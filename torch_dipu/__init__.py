import os

import torch

# use env to control?
mockcuda = True
from torch_dipu import _C
from torch_dipu import dipu 
from torch_dipu.dipu import *
from dipu.device import _get_device_index
from torch.serialization import register_package


def validate_dipu_device(location):
    device = _get_device_index(location, True)

    if not is_available():
        raise RuntimeError('Attempting to deserialize object on a DIPU '
                           'device but dipu is_available() is False. '
                           'If you are running on a CPU-only machine, '
                           'please use torch.load with map_location=torch.device(\'cpu\') '
                           'to map your storages to the CPU.')
    device_count = device_count()
    if device >= device_count:
        raise RuntimeError('Attempting to deserialize object on DIPU device '
                           f'{device} but dipu.device_count() is {device_count}. Please use '
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

    torch.Tensor.new_empty = GetDeviceProxy(torch.Tensor.new_empty,  pos = -1)
    torch.Tensor.new_empty_strided = GetDeviceProxy(torch.Tensor.new_empty_strided,  pos = -1)
    torch.Tensor.new_full = GetDeviceProxy(torch.Tensor.new_full,  pos = -1)
    torch.Tensor.new_ones = GetDeviceProxy(torch.Tensor.new_ones,  pos = -1)
    torch.Tensor.new_zeros = GetDeviceProxy(torch.Tensor.new_zeros,  pos = -1)
    # --- add other device func

    torch.Tensor.dipu = GetDeviceProxy(_C.dipu)
    torch.Tensor.is_dipu = GetDeviceProxy(_C.is_dipu)
    if mockcuda:
        torch.Tensor.cuda = torch.Tensor.dipu
        torch.Tensor.is_cuda = torch.Tensor.is_dipu


# mock device functions in generated/python_torch_functionsEverything.cpp
def apply_torch_function_patch():
    torch._C._nn._parse_to = GetDeviceProxy(torch._C._nn._parse_to, name = "no_name", static_func = True)
    torch.ones = GetTorchFuncProxy(torch.ones)
    torch.ones_like = GetTorchFuncProxy(torch.ones_like)
    torch.zeros = GetTorchFuncProxy(torch.zeros)
    torch.zeros_like = GetTorchFuncProxy(torch.zeros_like)

    torch.arange = GetTorchFuncProxy(torch.arange)
    torch.empty = GetTorchFuncProxy(torch.empty)
    torch.empty_like = GetTorchFuncProxy(torch.empty_like)
    torch.empty_strided = GetTorchFuncProxy(torch.empty_strided)

    torch.eye = GetTorchFuncProxy(torch.eye)
    torch.full = GetTorchFuncProxy(torch.full)
    torch.full_like = GetTorchFuncProxy(torch.full_like)
    torch.from_file = GetTorchFuncProxy(torch.from_file)
    torch._pin_memory = GetTorchFuncProxy(torch._pin_memory)
    torch.scalar_tensor = GetTorchFuncProxy(torch.scalar_tensor)

    torch.rand = GetTorchFuncProxy(torch.rand)
    torch.rand_like = GetTorchFuncProxy(torch.rand_like)
    torch.randint = GetTorchFuncProxy(torch.randint)
    torch.randint_like = GetTorchFuncProxy(torch.randint_like)
    torch.randn = GetTorchFuncProxy(torch.randn)
    torch.randn_like = GetTorchFuncProxy(torch.randn_like)
    torch.randperm = GetTorchFuncProxy(torch.randperm)
    if mockcuda:
        # torch.cuda = dipu
        torch.cuda.is_initialized = dipu.is_initialized
        torch.cuda.is_available = dipu.is_available
        torch.cuda.current_device = dipu.current_device
        torch.cuda.device = dipu.device
        torch.cuda.device_count = dipu.device_count
        torch.cuda.device_of = dipu.device_of
        torch.cuda.synchronize = dipu.synchronize
        torch.cuda.set_device = dipu.set_device
        torch.cuda.get_device_name = dipu.get_device_name

# temp solution, need redesign storage
def apply_temp_patch():
    from torch import Tensor
    _has_storage_raw = torch._C._has_storage
    def _has_storage_wrapper(x: Tensor):
        if x.device.type == "privateuseone":
            return False
        else:
            return _has_storage_raw(x)
    torch._C._has_storage = _has_storage_wrapper

def apply_patches():
    apply_tensor_method_patch()
    apply_torch_function_patch()
    apply_temp_patch()

apply_patches()
