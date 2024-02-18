import os
def patch_deepspeed():
    try:
        import torch
        import deepspeed
        import torch_dipu
        print("DIPU is patching deepspeed.")
        patch_cuda_accelerator_in_deepspeed()
    except ImportError:
        pass

def patch_cuda_accelerator_in_deepspeed():
    from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
    import torch
    if hasattr(CUDA_Accelerator,"is_fp16_supported"):
        CUDA_Accelerator.is_fp16_supported = custom_is_fp16_supported
    if hasattr(CUDA_Accelerator,"is_triton_supported"):
        CUDA_Accelerator.is_triton_supported = custom_is_triton_supported

def custom_is_fp16_supported(self):
    return True

def custom_is_triton_supported(self):
    return False


