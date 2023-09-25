# Copyright (c) 2023, DeepLink.
import os
# os.environ['DIPU_MEM_CHECK'] = '1'
# os.environ['DIPU_MEM_CHECK_MAX_BLOCK'] = '10000'
# os.environ['DIPU_MEM_CHECK_LOG_INTERVAL'] = '1000'
# os.environ['DIPU_MEM_CHECK_ENABLE_BACKTRACE'] = '1'

os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
mockcuda = False if os.environ.get("DIPU_MOCK_CUDA", 'True').lower()=='false' else True

import torch
from typing import (Tuple, List, Union, Sequence)
from torch.types import (_int, _size, Device, Number)
from torch import Tensor

# use env to control?
from torch_dipu import _C
from torch_dipu import dipu
from torch_dipu.dipu import *
from .dipu.distributed import apply_dist_patch
from .dipu.tensor import apply_tensor_type_patch
from .profiler.profiler import dipu_profiler, dipu_kineto_available
from .dipu.dataloader import apply_dataloader_patch
from .dipu.generator import apply_generator_patch
from .dipu.streams import apply_stream_patch, _dipu_record_stream

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
    torch.Tensor.is_dipu = property(_C.is_dipu)

    # if we replace in pybind layer, the func torch capacity in default python_variable_methods.cpp 
    # THPVariable_record_stream() will loss. so we currently replace in the python layer.
    torch.Tensor.record_stream = _dipu_record_stream

    if mockcuda:
        torch.Tensor.cuda = torch.Tensor.dipu
        torch.Tensor.is_cuda = torch.Tensor.is_dipu


# mock device functions in generated/python_torch_functionsEverything.cpp
def apply_torch_function_patch():
    torch._C._nn._parse_to = GetDeviceProxy(torch._C._nn._parse_to, caller = "static")
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

            if attr in torch.cuda.random.__all__ and hasattr(torch.cuda.random, attr):
                setattr(torch.cuda.random, attr, getattr(dipu, attr))


# temp solution, need redesign storage
def apply_temp_patch():
    def script_wrapper(*args, **kwargs):
        pass
    torch.jit.script = script_wrapper


def apply_profiler_patch():
    setattr(torch.profiler, 'kineto_available', dipu_kineto_available)
    setattr(torch.autograd.profiler, 'kineto_available', dipu_kineto_available)
    torch.profiler.profile = dipu_profiler


def apply_patches():
    apply_tensor_method_patch()
    apply_torch_function_patch()
    apply_dist_patch()
    apply_tensor_type_patch()
    apply_profiler_patch()
    apply_temp_patch()
    apply_dataloader_patch()
    apply_generator_patch()
    apply_stream_patch()

apply_patches()
