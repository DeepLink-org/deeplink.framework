# Copyright (c) 2023, DeepLink.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['MLU_INVOKE_BLOCKING'] = '1'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
# os.environ['DIPU_MEM_CHECK'] = '1'
# os.environ['DIPU_MEM_CHECK_MAX_BLOCK'] = '10000'
# os.environ['DIPU_MEM_CHECK_LOG_INTERVAL'] = '1000'
# os.environ['DIPU_MEM_CHECK_ENABLE_BACKTRACE'] = '1'
mockcuda = False if os.environ.get("DIPU_MOCK_CUDA", 'True').lower()=='false' else True

import torch
from typing import (Tuple, List, Union, Sequence)
from torch.types import (_int, _size, Device, Number)
from torch import Tensor

# use env to control?
from torch_dipu import _C
from torch_dipu import dipu 
from torch_dipu.dipu import *
from torch.serialization import register_package
from .dipu.device import _get_device_index
from .dipu.distributed import apply_dist_patch
from .dipu.tensor import apply_tensor_type_patch

def validate_dipu_device(location):
    device = _get_device_index(location, True)

    if not is_available():
        raise RuntimeError('Attempting to deserialize object on a DIPU '
                           'device but dipu is_available() is False. '
                           'If you are running on a CPU-only machine, '
                           'please use torch.load with map_location=torch.device(\'cpu\') '
                           'to map your storages to the CPU.')
    cnt = device_count()
    if device >= cnt:
        raise RuntimeError('Attempting to deserialize object on DIPU device '
                           f'{device} but dipu.device_count() is {cnt}. Please use '
                           'torch.load with map_location to map your storages '
                           'to an existing device.')
    return device

def _dipu_deserialize(obj, location):
    if location.startswith(dipu.diputype):
        device = validate_dipu_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            with dipu.device(device):
                return torch.UntypedStorage(obj.nbytes(), device=torch.device(location))
        else:
            return obj.dipu(device)

def _dipu_tag(obj):
    if obj.device.type == dipu.diputype:
        return dipu.diputype + str(obj.device.index)

# Tensor. _reduce_ex_internal  use numpy now
# register_package(30, _dipu_tag, _dipu_deserialize)

# mock device functions in generated/python_variable_methods.cpp 
def apply_tensor_method_patch():
    torch.Tensor.to = GetDeviceProxy(torch.Tensor.to)
    torch.Tensor.is_pinned = GetDeviceProxy(torch.Tensor.is_pinned)
    torch.Tensor.pin_memory = GetDeviceProxy(torch.Tensor.pin_memory)

    torch.Tensor.new_tensor = GetDeviceProxy(torch.Tensor.new_tensor,  pos = -1)
    torch.Tensor.new_empty = GetDeviceProxy(torch.Tensor.new_empty,  pos = -1)
    torch.Tensor.new_empty_strided = GetDeviceProxy(torch.Tensor.new_empty_strided,  pos = -1)
    torch.Tensor.new_full = GetDeviceProxy(torch.Tensor.new_full,  pos = -1)
    torch.Tensor.new_ones = GetDeviceProxy(torch.Tensor.new_ones,  pos = -1)
    torch.Tensor.new_zeros = GetDeviceProxy(torch.Tensor.new_zeros,  pos = -1)
    # --- add other device func
    # legacy api
    torch.Tensor.new =  GetDeviceProxy(torch.Tensor.new,  pos = -1)

    torch.Tensor.dipu = GetDeviceProxy(_C.dipu)
    torch.Tensor.is_dipu = GetDeviceProxy(_C.is_dipu)
    if mockcuda:
        torch.Tensor.cuda = torch.Tensor.dipu
        torch.Tensor.is_cuda = torch.Tensor.is_dipu


# mock device functions in generated/python_torch_functionsEverything.cpp
def apply_torch_function_patch():
    torch._C._nn._parse_to = GetDeviceProxy(torch._C._nn._parse_to, static_func = True)
    torch.ones = GetDeviceStaticProxy(torch.ones)
    torch.ones_like = GetDeviceStaticProxy(torch.ones_like)
    torch.zeros = GetDeviceStaticProxy(torch.zeros)
    torch.zeros_like = GetDeviceStaticProxy(torch.zeros_like)
    torch.as_tensor = GetDeviceStaticProxy( torch.as_tensor)
    torch.tensor = GetDeviceStaticProxy(torch.tensor)
    torch.arange = GetDeviceStaticProxy(torch.arange)
    torch.range = GetDeviceStaticProxy(torch.range)

    torch.empty = GetDeviceStaticProxy(torch.empty)
    torch.empty_like = GetDeviceStaticProxy(torch.empty_like)
    torch.empty_strided = GetDeviceStaticProxy(torch.empty_strided)

    torch.eye = GetDeviceStaticProxy(torch.eye)
    torch.full = GetDeviceStaticProxy(torch.full)
    torch.full_like = GetDeviceStaticProxy(torch.full_like)
    torch.from_file = GetDeviceStaticProxy(torch.from_file)
    torch._pin_memory = GetDeviceStaticProxy(torch._pin_memory)
    torch.scalar_tensor = GetDeviceStaticProxy(torch.scalar_tensor)

    torch.rand = GetDeviceStaticProxy(torch.rand)
    torch.rand_like = GetDeviceStaticProxy(torch.rand_like)
    torch.randint = GetDeviceStaticProxy(torch.randint)
    torch.randint_like = GetDeviceStaticProxy(torch.randint_like)
    torch.randn = GetDeviceStaticProxy(torch.randn)
    torch.randn_like = GetDeviceStaticProxy(torch.randn_like)
    torch.randperm = GetDeviceStaticProxy(torch.randperm)
    if mockcuda:
        for attr in dipu.__all__:
            if hasattr(torch.cuda, attr):
                setattr(torch.cuda, attr, getattr(dipu, attr))


# temp solution, need redesign storage
def apply_temp_patch():
    _has_storage_raw = torch._C._has_storage
    def _has_storage_wrapper(x: Tensor):
        if x.device.type == diputype:
            return False
        else:
            return _has_storage_raw(x)
    torch._C._has_storage = _has_storage_wrapper

    def script_wrapper(*args, **kwargs):
        pass
    torch.jit.script = script_wrapper


def apply_patches():
    apply_tensor_method_patch()
    apply_torch_function_patch()
    apply_temp_patch()
    apply_dist_patch()
    # apply_tensor_type_patch()


apply_patches()
