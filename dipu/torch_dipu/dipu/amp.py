import threading

import torch

from torch_dipu import _C, dipu

thread_local = threading.local()


def init_autocast_dtype():
    if not hasattr(thread_local, "inited"):
        thread_local.inited = 1
        set_autocast_dipu_dtype(torch.float16)


def get_autocast_dipu_dtype():
    # autocast_xpu_dtype is defined in ATen/autocast_mode.cpp with default value kBFloat16,
    # however we want it to be kHalf just as cuda,
    # while it is a thread local value so change its value at the first time when user getting autocast dtype in each thread
    init_autocast_dtype()
    return _C.get_autocast_dipu_dtype()


def is_autocast_dipu_enabled():
    return _C.is_autocast_dipu_enabled()


def set_autocast_dipu_enabled(enabled):
    return _C.set_autocast_dipu_enabled(enabled)


def set_autocast_dipu_dtype(dtype):
    # if user have set autocast dtype, there is no need to change its default value anymore
    thread_local.inited = 1
    return _C.set_autocast_dipu_dtype(dtype)


# bf16 is not supported by default.
# This function needs to be improved in the future and customized for different device.
def is_bf16_supported():
    return False


def apply_amp_patch():
    torch.get_autocast_gpu_dtype = get_autocast_dipu_dtype
    torch.set_autocast_gpu_dtype = set_autocast_dipu_dtype
    torch.set_autocast_enabled = set_autocast_dipu_enabled
    torch.is_autocast_enabled = is_autocast_dipu_enabled
    # If vendor is cuda, its ability to support bf16 remains the same as the default.
    # (which depends on the Compute Capability)
    if (dipu.vendor_type != "CUDA"):
        torch.cuda.is_bf16_supported = is_bf16_supported
