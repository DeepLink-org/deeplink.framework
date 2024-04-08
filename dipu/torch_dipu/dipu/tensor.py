# Copyright (c) 2023, DeepLink.
import torch

from .device import __diputype__, __dipu_device_type__
from torch_dipu import _C, mockcuda


_default_tensor_type = torch.FloatTensor


def __set_default_tensor_type(type=torch.FloatTensor):
    print(
        " warnning!! dipu not support default tensor setting now!, this func is empty"
    )
    global _default_tensor_type
    _default_tensor_type = type


_raw_tensor_type = torch.Tensor.type


def _wrap_tensor_type(self, *args, **kwargs):
    ret = _raw_tensor_type(self, *args, **kwargs)
    if isinstance(ret, str):
        return ret.replace(__dipu_device_type__, "cuda")
    else:
        return ret


# need enhance, seems change tensor define is need
def apply_tensor_type_patch():
    torch.set_default_tensor_type = __set_default_tensor_type
    if mockcuda:
        _C._mockCudaTensor()
    torch.Tensor.type = _wrap_tensor_type
