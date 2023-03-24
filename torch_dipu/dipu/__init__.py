import torch
from torch_dipu import mockcuda
from torch_dipu import _C
from .device import __diputype__ as diputype
from .device import __vendor__ as vendor_type
from .device import _DeviceWrapper
from .device import *
from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
                    synchronize, device_count, set_device, current_device, get_device_name,
                    get_device_properties, _get_device_index, is_available, device, device_of,
                    stream, current_stream, default_stream)
from .random_dipu import manual_seed, manual_seed_all, seed, seed_all, initial_seed
from .memory import (_free_mutex, caching_allocator_alloc, caching_allocator_delete,
                     empty_cache)
from .streams import Stream, Event

__all__ = [
    "device", "vendor_device", "is_initialized", "_lazy_call", "_lazy_init", "init",
    "synchronize", "device_count", "set_device", "current_device", "get_device_name",
    "get_device_properties", "_get_device_index", "is_available", "device", "device_of",
    "stream", "current_stream", "default_stream", "manual_seed", "manual_seed_all",
    "seed", "seed_all", "initial_seed", "_free_mutex", "caching_allocator_alloc",
    "caching_allocator_delete", "empty_cache", "Stream", "Event", "mockcuda"
]


def apply_rt_patch():
    torch.device = _DeviceWrapper

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
    torch._C._nn._parse_to = GetDeviceProxy(torch._C._nn._parse_to, static_func = True)

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

def apply_patches():
    apply_rt_patch()
    apply_tensor_method_patch()
    apply_torch_function_patch()


apply_patches()
