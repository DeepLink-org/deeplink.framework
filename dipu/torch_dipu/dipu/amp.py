from torch_dipu import _C, dipu
import torch


def get_autocast_dipu_dtype():
    return _C.get_autocast_dipu_dtype()


def is_autocast_dipu_enabled():
    return _C.is_autocast_dipu_enabled()


def set_autocast_dipu_enabled(enabled):
    return _C.set_autocast_dipu_enabled(enabled)


def set_autocast_dipu_dtype(dtype):
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

    # autocast_xpu_dtype is defined in ATen/autocast_mode.cpp with default value kBFloat16.
    # it is a thread local so this set only change default value in main thread.
    # ** need enhance to let all threads has same default type.
    set_autocast_dipu_dtype(torch.float16)