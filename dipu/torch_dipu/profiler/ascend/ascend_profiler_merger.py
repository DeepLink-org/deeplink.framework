import json
import os
import subprocess
from typing import Union
from decimal import Decimal


class _PathManager:
    @staticmethod
    def _remove_directory(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist")
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        os.rmdir(path)

    @staticmethod
    def get_npu_profile_path(tmp_file_path: str) -> str:
        # in the correct case, there is only one directory in the folder
        for dir in os.listdir(tmp_file_path):
            return os.path.join(tmp_file_path, dir)
        return ""

    @staticmethod
    def get_msprof_profile_json_path(npu_profile_path: str) -> list:
        msprof_profile_path = os.path.join(
            npu_profile_path, "mindstudio_profiler_output"
        )
        msprof_profile_json_path_list = []
        for data_file in os.listdir(msprof_profile_path):
            if data_file.startswith("msprof_"):
                msprof_profile_json_path_list.append(
                    os.path.join(msprof_profile_path, data_file)
                )
        return msprof_profile_json_path_list

    @classmethod
    def remove_temp_msprof_directory(cls, path: str) -> None:
        if "/tmp/aclprof" not in path:
            raise ValueError("Invalid temp msprof path format")
        cls._remove_directory(path)


# the main process of this class is as follows:
# 1.load and preprocess kineto events:
# get each process's name and id
# get python3 process sort index,
# get temp file path and torch time diff
# 2.load and merge(if msprof json is splited) msprof profile data:
# filter unnecessary cann events
# get acl time diff
# 3.adjust flow event HostToDevice's timestamp
# 4.merge msprof data to kineto data
# 5.export json file
class _AscendProfilerMerger:
    def __init__(self, kineto_profile_json_path: str):
        self._kineto_profile_data = self._load_chrome_trace_json(
            kineto_profile_json_path
        )
        self._msprof_profile_data = []
        self._process_id = {
            "python3": None,
            "CANN": None,
            "Ascend Hardware": None,
            "Overlap Analysis": None,
            "HCCL": None,
            "AI Core Freq": None,
        }

        # this sort_idx is used to make sure python3 is above hardware event
        self._python3_sort_idx = 0

        # time diff between realtime and monotonic_raw
        self._acl_time_diff = 0
        self._torch_time_diff = 0

        # to store beforehand sorted cann_x_event for the binary search to align start flow event to cann event
        self._msprof_cann_x_event = []

        self._hardware_process_list = [
            "Ascend Hardware",
            "Overlap Analysis",
            "HCCL",
            "AI Core Freq",
        ]

        self._export_path = kineto_profile_json_path

        self._preprocess_kineto_profile_data()

    # in the correct case, kineto_profile_data is a dict
    # and msprof_profile_data is a list
    @staticmethod
    def _load_chrome_trace_json(file_path: str) -> Union[dict, list]:
        with open(file_path, "r") as file:
            return json.load(file)

    def _preprocess_kineto_profile_data(self) -> None:

        # find and delete special events passed by cpp which represent for npu profile dump path
        # in the correct case, one msprof.json file corresponds to one Kineto.json file.
        temp_path_event = {}
        torch_time_diff_event = {}

        temp_path_event_begin_name = "random_temp_dir:"
        torch_time_diff_event_begin_name = "torch_time_diff:"

        # get each process's name, id and sort_index
        # get special event that for temp_path and torch_time_diff
        for event in self._kineto_profile_data["traceEvents"]:
            if event["name"] == "process_name" and self._process_id["python3"] == None:
                self._process_id["python3"] = event["pid"]
            if (
                event["name"] == "process_sort_index"
                and event["pid"] == self._process_id["python3"]
            ):
                self._python3_sort_idx = event["args"]["sort_index"]
            if (
                "cat" in event.keys()
                and event["cat"] == "user_annotation"
                and event["name"].startswith(temp_path_event_begin_name)
            ):
                temp_path_event = event
            elif event["name"].startswith(torch_time_diff_event_begin_name):
                torch_time_diff_event = event
        self._kineto_profile_data["traceEvents"].remove(temp_path_event)
        self._kineto_profile_data["traceEvents"].remove(torch_time_diff_event)

        npu_temp_path = temp_path_event["name"][len(temp_path_event_begin_name) :]
        self._torch_time_diff = int(
            torch_time_diff_event["name"][len(torch_time_diff_event_begin_name) :]
        )
        npu_profile_path = _PathManager.get_npu_profile_path(npu_temp_path)
        command = ["msprof", "--export=on", f"--output={npu_profile_path}/."]

        # the existence of the msprof command is ensured before class initialization
        subprocess.run(command, capture_output=True, text=True)
        msprof_profile_json_path_list = _PathManager.get_msprof_profile_json_path(
            npu_profile_path
        )
        for msprof_profile_json_path in msprof_profile_json_path_list:
            if self._msprof_profile_data == []:
                self._filter_msprof_profile_event(
                    self._load_chrome_trace_json(msprof_profile_json_path)
                )
            else:
                self._merge_msprof_profile_data(
                    self._load_chrome_trace_json(msprof_profile_json_path)
                )

        # get acl_time_diff
        msprof_end_info_path = os.path.join(npu_profile_path, "host/end_info")
        with open(msprof_end_info_path, "r") as file:
            end_info = json.load(file)
        self._acl_time_diff = int(end_info["collectionTimeEnd"]) * 1000 - int(
            end_info["clockMonotonicRaw"]
        )

        # _PathManager.remove_temp_msprof_directory(npu_temp_path)

    # imitate torch_npu to filter out certain CANN layers' events that are not needed to be displayed
    # to improve efficiency, also count the relationship between process_id and process_name while filtering
    def _filter_msprof_profile_event(self, input_msprof_data: list) -> None:
        def filter_event_condition(event: dict) -> bool:
            if event["ph"] == "M":
                return False
            if event["pid"] != self._process_id["CANN"]:
                return False
            if event["name"].startswith("HostToDevice"):
                return False
            if event["name"].startswith("AscendCL"):
                if event["args"]["id"].startswith("acl"):
                    return False
            return True

        # the event arrangement in the msprof JSON file has a special property,
        # that the "process_name" event is fixed at the beginning,
        # so two tasks can be optimized by iterating once
        # if the arrangement changes,
        # the following code will no longer be applicable,
        # and needs to be split into two iteration
        for event in input_msprof_data:
            if filter_event_condition(event):
                continue
            if event["name"] == "process_name":
                self._process_id[event["args"]["name"]] = event["pid"]
            self._msprof_profile_data.append(event)
            if event["ph"] == "X" and event["pid"] == self._process_id["CANN"]:
                self._msprof_cann_x_event.append(
                    (Decimal(event["ts"]), Decimal(event["ts"]) + Decimal(event["dur"]))
                )

    def _merge_msprof_profile_data(self, input_msprof_data: list) -> None:
        for event in input_msprof_data:
            if (
                event["ph"] != "X"
                and event["ph"] != "f"
                and event["ph"] != "s"
                and event["ph"] != "C"
            ):
                continue
            self._msprof_profile_data.append(event)

    # this function has two main things to do:
    # 1.binary search each HostToDevice flow start event to align it
    # 2.calculate and apply hardware event offset to make sure arrow will be displayed normally
    def _adjust_HostToDevice_event_offset(self) -> None:
        # align HostToDevice's timestamp to its cann_x_event's begin timestamp
        # msprof_cann_x_event is sorted by timestamp beforehand,
        # and what we need to do is binary search for the event whose time interval contain this timestamp
        # and move start flow event HostToDevice's timestamp to the cann event's begin timestamp
        def find_wrap_acl_event(ts: float) -> float:
            if len(self._msprof_cann_x_event) == 0:
                return ts
            l = 0
            r = len(self._msprof_cann_x_event) - 1
            while l < r:
                mid = int((l + r + 1) / 2)
                if self._msprof_cann_x_event[mid][0] < ts:
                    l = mid
                else:
                    r = mid - 1
            if (
                self._msprof_cann_x_event[l][0] < ts
                and self._msprof_cann_x_event[l][1] > ts
            ):
                return self._msprof_cann_x_event[l][0]
            else:
                return ts

        flow_start_dict = {}
        flow_end_dict = {}

        for event in self._msprof_profile_data:
            if not event["name"].startswith("HostToDevice"):
                continue
            if event["pid"] == self._process_id["CANN"]:
                event["ts"] = find_wrap_acl_event(float(event["ts"]))
                flow_start_dict[event["id"]] = event["ts"]
            elif event["pid"] == self._process_id["Ascend Hardware"]:
                flow_end_dict[event["id"]] = event["ts"]

        # the minimum hardware offset to make sure HostToDevice start event's timestamp is smaller than HostToDevice end event's
        ts_min_offset_need = 0
        for key in flow_start_dict.keys():
            if key in flow_end_dict.keys():

                # hardware offset needed for current event
                # for example if end time is 2 and start time is 4
                # we need to offset event in hardware (start time - end time) + 1 = 4 - 2 + 1 = 3
                # after offset, the end time is going to be 2 + 3 = 5 which is greater than 4
                # and for all results, we need to get the maximal value among them
                current_ts_offset_need = float(flow_start_dict[key]) - float(
                    flow_end_dict[key]
                )
                ts_min_offset_need = max(ts_min_offset_need, current_ts_offset_need + 1)
        if ts_min_offset_need == 0:
            return
        for event in self._msprof_profile_data:
            if "ts" not in event.keys():
                continue
            for hardware_process_name in self._hardware_process_list:
                if event["pid"] == self._process_id[hardware_process_name]:
                    event["ts"] = str(float(event["ts"]) + ts_min_offset_need)
                    break

    def _merge_msprof_to_kineto(self) -> None:
        # to make sure cann timestamp align to kineto timestamp
        local_time_diff = Decimal((self._torch_time_diff - self._acl_time_diff) / 1000)

        for event in self._msprof_profile_data:
            if event["pid"] == self._process_id["CANN"]:
                event["pid"] = self._process_id["python3"]
            if event["name"] == "process_sort_index":
                event["args"]["sort_index"] += self._python3_sort_idx
            if "ts" in event.keys():
                event["ts"] = str(Decimal(event["ts"]) + local_time_diff)
            self._kineto_profile_data["traceEvents"].append(event)

    def _export_to_json(self) -> None:
        # if generated json is too large, we will compress its output format
        MAX_EVENTS_TO_COMPRESS = 100000
        with open(self._export_path, "w") as json_file:
            if len(self._kineto_profile_data["traceEvents"]) > MAX_EVENTS_TO_COMPRESS:
                json.dump(self._kineto_profile_data, json_file, separators=(",", ":"))
            else:
                json.dump(self._kineto_profile_data, json_file, indent=1)

    def start_merge(self) -> None:
        self._adjust_HostToDevice_event_offset()
        self._merge_msprof_to_kineto()
        self._export_to_json()


def _command_exists(command) -> bool:
    try:
        subprocess.run(["which", command], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("msprof command not exist, merge process will be canceled")
        return False


def merge_kineto_and_msprof_profile_data(path: str) -> None:
    if not _command_exists("msprof"):
        return
    merger = _AscendProfilerMerger(path)
    merger.start_merge()
