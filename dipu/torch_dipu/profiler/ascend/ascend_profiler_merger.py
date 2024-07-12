import json
import os
import subprocess
from typing import Union, Tuple

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
        # 正常情况下，文件夹内有且仅有一个dir
        for dir in os.listdir(tmp_file_path):
            return os.path.join(tmp_file_path, dir)
        return ""
    
    @staticmethod
    def get_msprof_profile_json_path(npu_profile_path: str) -> Union[str, None]:
        msprof_profile_path = os.path.join(npu_profile_path, 'mindstudio_profiler_output')
        for data_file in os.listdir(msprof_profile_path):
            if data_file.startswith('msprof_'):
                return os.path.join(msprof_profile_path, data_file)
        return None

    @classmethod
    def remove_temp_msprof_directory(cls, path: str) -> None:
        if path.startswith('/tmp/aclprof') == False:
            raise ValueError("Invalid temp msprof path format")
        cls._remove_directory(path)
        
class _AscendProfilerMerger:
    def __init__(self, kineto_profile_json_path: str):
        self._kineto_profile_data = self._load_chrome_trace_json(kineto_profile_json_path)
        self._msprof_profile_data = []

        # 每一层process对应的process_id
        self._process_id = {
            'python3': None,
            'CANN': None,
            'Ascend Hardware': None,
            'Overlap Analysis': None,
            'HCCL': None,
            'AI Core Freq': None
        }

        # 保证python层早于hardware层显示
        self._python3_sort_idx = 0

        # 保证所有python to kernel的箭头都可以被连上的最小ts_offset
        self._ts_min_offset = 0.0

        # 整体偏移的process key（硬件层）
        self._hardware_process_list = ['Ascend Hardware', 'Overlap Analysis', 'HCCL', 'AI Core Freq']

        self._export_path = kineto_profile_json_path

        self._preprocess_kineto_profile_data()

    # 目前，kineto_profile_data是dict类型
    # msprof_profile_data是list类型
    @staticmethod
    def _load_chrome_trace_json(file_path: str) -> Union[dict, list]:
        with open(file_path, 'r') as file:
            return json.load(file)

    def _preprocess_kineto_profile_data(self) -> None:

        # 找到并删除C++层传上来的特殊event（代表npu profile路径）
        temp_path_event_list = []
        for event in self._kineto_profile_data['traceEvents']:
            if event['name'] == 'process_name' and self._process_id['python3'] == None:
                self._process_id['python3'] = event['pid']
            if 'cat' in event.keys() and event['cat'] == 'user_annotation' and event['name'].startswith('random_temp_dir'):
                temp_path_event_list.append(event)
            if event['name'] == 'process_sort_index' and event['pid'] == self._process_id['python3']:
                self._python3_sort_idx = event['args']['sort_index']
        self._kineto_profile_data['traceEvents'] = [
            event for event in self._kineto_profile_data['traceEvents'] if event not in temp_path_event_list]

        npu_temp_path_count = set()
        for temp_path_event in temp_path_event_list:
            npu_temp_path = temp_path_event['name'][16:]
            if npu_temp_path in npu_temp_path_count:
                continue

            npu_temp_path_count.add(npu_temp_path)
            # 提取C++层传上来的temp路径
            npu_profile_path = _PathManager.get_npu_profile_path(npu_temp_path)
            command = ["msprof", "--export=on", f"--output={npu_profile_path}/."] 

            # 在类初始化前确保了msprof命令存在
            subprocess.run(command, capture_output=True, text=True)
            msprof_profile_json_path = _PathManager.get_msprof_profile_json_path(npu_profile_path)
            if self._msprof_profile_data == []:
              self._process_id, self._msprof_profile_data = self._filter_msprof_profile_event(self._load_chrome_trace_json(msprof_profile_json_path))
            else:
              self._merge_msprof_profile_data(self._load_chrome_trace_json(msprof_profile_json_path))

            # 删除npu profile临时文件
            _PathManager.remove_temp_msprof_directory(npu_temp_path)

    # 模仿torch_npu，将某些cann层不需要被显示的event过滤掉
    # 为了提高效率，在过滤的同时统计process_id和process_name之间的关系
    def _filter_msprof_profile_event(self, input_msprof_data: list) -> Tuple[dict, list]:
        msprof_process_id = self._process_id
        filtered_msprof_event_list = []
        def filter_event_condition(event: dict) -> bool:
            if event['pid'] != msprof_process_id['CANN']:
                return False
            if event['name'].startswith('HostToDevice'):
                return False
            if event['name'].startswith('AscendCL'):
                if event['args']['id'].startswith('acl'):
                    return False
            return True

        # 按照msprof排列，process_name event固定在开头位置，因此两个任务可以优化成一次遍历
        # 若排列顺序发生变化，本部分代码将不再适用，需要拆分成两个遍历
        for event in input_msprof_data:
            if event['name'] == 'process_name':
                for process_name in self._process_id.keys():
                    if event['args']['name'] == process_name:
                        msprof_process_id[process_name] = event['pid']
            elif filter_event_condition(event) == True:
                continue
            filtered_msprof_event_list.append(event)

        return msprof_process_id, filtered_msprof_event_list
    
    # 将底层产生的多个msprof json文件合并
    def _merge_msprof_profile_data(self, input_msprof_data: list) -> None:
        process_id, input_msprof_data = self._filter_msprof_profile_event(input_msprof_data)
        for event in input_msprof_data:
            if event['ph'] != 'X' and event['ph'] != 'f' and event['ph'] != 's':
                continue
            for key in process_id.keys():
                if event['pid'] != process_id[key]:
                    continue
                event['pid'] = self._process_id[key]
                self._msprof_profile_data.append(event)
                break
    
    def _calculate_ts_min_offset(self) -> None:
        flow_start_dict = {}
        flow_end_dict = {}
        for event in self._msprof_profile_data:
            if not event['name'].startswith('HostToDevice'):
                continue
            if event['pid'] == self._process_id['CANN']:
                flow_start_dict[event['id']] = event['ts']
            elif event['pid'] == self._process_id['Ascend Hardware']:
                flow_end_dict[event['id']] = event['ts']

        for key, value in flow_start_dict.items():
            if key in flow_end_dict.keys():
                current_ts_offset_need = float(value) - float(flow_end_dict[key])
                self._ts_min_offset = max(self._ts_min_offset, current_ts_offset_need + 1)
    
    def _add_ts_offset_to_hardware(self) -> None:
        for event in self._msprof_profile_data:
            if 'ts' not in event.keys():
                continue
            for hardware_process_name in self._hardware_process_list:
                if event['pid'] == self._process_id[hardware_process_name]:
                    event['ts'] = str(float(event['ts']) + self._ts_min_offset)  
                    break
    
    def _merge_msprof_to_kineto(self) -> None:
        for event in self._kineto_profile_data['traceEvents']:
            if event['name'] == 'process_name':
                self._process_id['python3'] = event['pid']
                break
        
        for event in self._msprof_profile_data:
            if event['pid'] == self._process_id['CANN']:
                event['pid'] = self._process_id['python3']
            if event['name'] == 'process_sort_index':
                event['args']['sort_index'] += self._python3_sort_idx
            self._kineto_profile_data['traceEvents'].append(event)
        
    def _export_to_json(self) -> None:
        with open(self._export_path, 'w') as json_file:
            json.dump(self._kineto_profile_data, json_file, indent=4)
    
    def start_merge(self) -> None:
        self._calculate_ts_min_offset()
        self._add_ts_offset_to_hardware()
        self._merge_msprof_to_kineto()
        self._export_to_json()

def _command_exists(command) -> bool:
    try:
        subprocess.run(['which', command], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("msprof command not exist, merge process will be canceled")
        return False
    
def merge_kineto_and_msprof_profile_data(path: str) -> None:
    if not _command_exists('msprof'):
        return
    merger = _AscendProfilerMerger(path)
    merger.start_merge()