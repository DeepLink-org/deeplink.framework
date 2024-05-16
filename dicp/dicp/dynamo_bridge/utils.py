import copy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch.fx
from torch._inductor.codecache import code_hash
from torch.fx.node import Argument, Target


def not_all_num_shape(shape):
    for elem in shape:
        if not isinstance(elem, int):
            return True
    return False


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def neg_in_shape(shape):
    for elem in shape:
        if isinstance(elem, int):
            if elem < 0:
                return True
    return False


def process_sym_name(st):
    # dynamic shape feature
    # return string wrapper in new version
    # node.str() will not fallback SymInt value form
    if isinstance(st, torch.SymInt):
        return st.node.str()
    return str(st)


def preprocess_expression(expr):
    elem_str = process_sym_name(expr)
    elem_str = elem_str.replace('+', ' + ')
    elem_str = elem_str.replace('-', ' - ')
    elem_str = elem_str.replace('*', ' * ')
    elem_str = elem_str.replace('//', ' // ')
    elem_str = elem_str.replace('(', ' ( ')
    elem_str = elem_str.replace(')', ' ) ')
    elems = elem_str.split(' ')
    elems = [e for e in elems if e != '']
    return elems


def find_root_num(set_num, num):
    while set_num[num] != num:
        num = set_num[num]
    return num


def merge_disjoint_set(set_num, idx_a, idx_b):
    root_a = find_root_num(set_num, idx_a)
    root_b = find_root_num(set_num, idx_b)
    # an example for (s5 / 8) - (s5 / 16)
    # num: 0 1 2 3
    # step1 - > set_num: 0 1 2 3
    # step2 - > set_num: 0 0 2 2
    # step3 - > set_num: 0 0 0 0

    # return merged set from root_b to root_a
    return [root_a if find_root_num(set_num, s) == root_b else s for s in set_num]


def save_cpu_gm(gm: torch.fx.GraphModule, folder: str):
    Path(folder).mkdir(exist_ok=True)
    cpu_gm = copy_gm_to_cpu(gm)
    grap_code = cpu_gm.code
    graph_key = code_hash(grap_code)
    cpu_gm.to_folder(folder + "/" + graph_key[:4], module_name=graph_key)
    return cpu_gm, graph_key


def copy_gm_to_cpu(gm: torch.fx.GraphModule):
    cpu_gm = copy.deepcopy(gm).cpu()
    return DeviceParamToCpu(cpu_gm).transform()


def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    elif tensor.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    else:
        return torch.contiguous_format


def get_cast_dtype(
    type1: Union[str, torch.dtype, type], type2: Union[str, torch.dtype, type]
) -> Union[str, torch.dtype, None]:
    if type1 == type2:
        return type1

    type_map = {
        int: torch.int,
        float: torch.float,
        complex: torch.complex,
        bool: torch.bool,
    }

    type1 = torch.dtype(type1) if isinstance(type1, str) else type1
    type2 = torch.dtype(type2) if isinstance(type2, str) else type2

    type1 = type_map[type1] if isinstance(type1, type) else type1
    type2 = type_map[type2] if isinstance(type2, type) else type2

    if type1 == torch.bool or type2 == torch.bool:
        return torch.bool
    elif type1 == torch.double or type2 == torch.double:
        return torch.double

    complex_list = [torch.complex32, torch.complex64, torch.complex128]
    float_list = [torch.float16, torch.float32, torch.float, torch.float64]
    int_list = [torch.int8, torch.int16, torch.int32, torch.int, torch.int64]

    if type1 in complex_list or type2 in complex_list:
        t1_idx = complex_list.index(type1) if type1 in complex_list else -1
        t2_idx = complex_list.index(type2) if type2 in complex_list else -1
        return complex_list[max(t1_idx, t2_idx)]
    elif type1 in float_list or type2 in float_list:
        t1_idx = float_list.index(type1) if type1 in float_list else -1
        t2_idx = float_list.index(type2) if type2 in float_list else -1
        return float_list[max(t1_idx, t2_idx)]
    elif type1 in int_list or type2 in int_list:
        t1_idx = int_list.index(type1) if type1 in int_list else -1
        t2_idx = int_list.index(type2) if type2 in int_list else -1
        return int_list[max(t1_idx, t2_idx)]

    assert False, str(type1) + " " + str(type2) + " can't cast these two types!"


class TensorInfo:
    def __init__(self, shape: list, dtype: torch.dtype, memory_format: torch.memory_format) -> None:
        self.shape = shape
        self.dtype = dtype
        self.memory_format = memory_format


class DeviceParamToCpu(torch.fx.Transformer):
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if 'device' in kwargs:
            new_kwargs = {k: v for k, v in kwargs.items()}
            new_kwargs['device'] = torch.device("cpu")
        else:
            new_kwargs = kwargs
        return super().call_function(target, args, new_kwargs)
