import copy
from typing import Any, Dict, Tuple, Mapping, Optional

import torch.fx
from torch._inductor.codecache import code_hash
from torch.fx.node import Argument, Target, Node
from torch.fx.passes.operator_support import OperatorSupport, SupportDict
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS


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
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    elif tensor.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    else:
        return torch.contiguous_format


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

class AotOperatorUnsupport(OperatorSupport):
    def __init__(self, unsupport_dict: Optional[SupportDict] = None, prefix=""):
        super().__init__(unsupport_dict)
        for key, value in list(self._support_dict.items()):
            if prefix not in key:
                del self._support_dict[key]
                self._support_dict[prefix + key] = value

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: Node
    ) -> bool:
        return True if node.op == "get_attr" else \
            not super().is_node_supported(submodules, node)
