import json
import os
import subprocess
from typing import Union


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
    def get_msprof_profile_json_path(npu_profile_path: str) -> Union[str, None]:
        msprof_profile_path = os.path.join(
            npu_profile_path, "mindstudio_profiler_output"
        )
        for data_file in os.listdir(msprof_profile_path):
            if data_file.startswith("msprof_"):
                return os.path.join(msprof_profile_path, data_file)
        return None

    @classmethod
    def remove_temp_msprof_directory(cls, path: str) -> None:
        if "/tmp/aclprof" not in path:
            raise ValueError("Invalid temp msprof path format")
        cls._remove_directory(path)


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

        # the minimum timestamp_offset that ensure every flow event could be displayed normally
        self._ts_min_offset = 0.0

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
        msprof_profile_json_path = _PathManager.get_msprof_profile_json_path(
            npu_profile_path
        )
        self._filter_msprof_profile_event(
            self._load_chrome_trace_json(msprof_profile_json_path)
        )

        # get acl_time_diff
        msprof_end_info_path = os.path.join(npu_profile_path, "host/end_info")
        with open(msprof_end_info_path, "r") as file:
            end_info = json.load(file)
        self._acl_time_diff = int(end_info["collectionTimeEnd"]) * 1000 - int(
            end_info["clockMonotonicRaw"]
        )

        _PathManager.remove_temp_msprof_directory(npu_temp_path)

    # imitate torch_npu to filter out certain CANN layers' events that are not needed to be displayed
    # to improve efficiency, also count the relationship between process_id and process_name while filtering
    def _filter_msprof_profile_event(self, input_msprof_data: list) -> None:
        def filter_event_condition(event: dict) -> bool:
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
            if event["name"] == "process_name":
                for process_name in self._process_id.keys():
                    if event["args"]["name"] == process_name:
                        self._process_id[process_name] = event["pid"]
            elif filter_event_condition(event) == True:
                continue
            self._msprof_profile_data.append(event)

    # make sure that every flow event's begin time is smaller than end time
    def _calculate_ts_min_offset(self) -> None:
        flow_start_dict = {}
        flow_end_dict = {}
        for event in self._msprof_profile_data:
            if not event["name"].startswith("HostToDevice"):
                continue
            if event["pid"] == self._process_id["CANN"]:
                flow_start_dict[event["id"]] = event["ts"]
            elif event["pid"] == self._process_id["Ascend Hardware"]:
                flow_end_dict[event["id"]] = event["ts"]

        for key in flow_start_dict.keys():
            if key in flow_end_dict.keys():

                # hardware offset needed for current event
                # for example if end time is 2 and start time is 4
                # we need to offset event in hardware (start time - end time) + 1
                # and for all results, we need to get the maximal value among them
                current_ts_offset_need = float(flow_start_dict[key]) - float(
                    flow_end_dict[key]
                )
                self._ts_min_offset = max(
                    self._ts_min_offset, current_ts_offset_need + 1
                )

    def _add_ts_offset_to_hardware(self) -> None:
        for event in self._msprof_profile_data:
            if "ts" not in event.keys():
                continue
            for hardware_process_name in self._hardware_process_list:
                if event["pid"] == self._process_id[hardware_process_name]:
                    event["ts"] = str(float(event["ts"]) + self._ts_min_offset)
                    break

    def _merge_msprof_to_kineto(self) -> None:
        for event in self._kineto_profile_data["traceEvents"]:
            if event["name"] == "process_name":
                self._process_id["python3"] = event["pid"]
                break

        # to make sure cann timestamp align to kineto timestamp
        local_time_diff = (self._torch_time_diff - self._acl_time_diff) / 1000

        for event in self._msprof_profile_data:
            if event["pid"] == self._process_id["CANN"]:
                event["pid"] = self._process_id["python3"]
            if event["name"] == "process_sort_index":
                event["args"]["sort_index"] += self._python3_sort_idx
            if "ts" in event.keys():
                event["ts"] = str(float(event["ts"]) + local_time_diff)
            self._kineto_profile_data["traceEvents"].append(event)

    def _export_to_json(self) -> None:
        with open(self._export_path, "w") as json_file:
            json.dump(self._kineto_profile_data, json_file, indent=4)

    def start_merge(self) -> None:
        self._calculate_ts_min_offset()
        self._add_ts_offset_to_hardware()
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
