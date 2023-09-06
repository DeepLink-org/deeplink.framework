import torch
from torch_dipu import _C


def get_autocast_xpu_dtype():
    return _C.get_autocast_xpu_dtype()

def is_autocast_xpu_enabled():
    return _C.is_autocast_xpu_enabled()

def set_autocast_xpu_enabled(enabled):
    return _C.set_autocast_xpu_enabled(enabled)

def set_autocast_xpu_dtype(dtype):
    return _C.set_autocast_xpu_dtype(dtype)

