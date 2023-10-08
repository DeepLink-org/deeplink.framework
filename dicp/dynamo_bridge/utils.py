import os
import copy
from typing import Any, Dict, Tuple

import torch.fx
from torch._inductor.codecache import code_hash
from torch.fx.node import Argument, Target


def save_cpu_gm(gm: torch.fx.GraphModule, folder: str):
    cpu_gm = copy_gm_to_cpu(gm)
    grap_code = cpu_gm.code
    graph_key = code_hash(grap_code)
    cpu_gm.to_folder(folder + "/" + graph_key[:4], module_name=graph_key)
    return cpu_gm, graph_key
    
def copy_gm_to_cpu(gm: torch.fx.GraphModule):
    cpu_gm = copy.deepcopy(gm).cpu()
    return DeviceParamToCpu(cpu_gm).transform()

def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format = torch.channels_last):
        return torch.channels_last
    elif tensor.is_contiguous(memory_format = torch.channels_last_3d):
        return torch.channels_last_3d
    else:
        return torch.contiguous_format

class TensorInfo:
    def __init__(self, shape:list, dtype:torch.dtype, memory_format:torch.memory_format) -> None:
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
