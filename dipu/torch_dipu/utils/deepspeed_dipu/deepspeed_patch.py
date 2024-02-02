import os
def patch_deepspeed():
    try:
        import torch
        import deepspeed
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
    if hasattr(torch.cuda,"nvtx") and hasattr(torch.cuda.nvtx,"range_push"):
        torch.cuda.nvtx.range_push = custom_range_push
    if hasattr(torch.cuda,"nvtx") and hasattr(torch.cuda.nvtx,"range_pop"):
        torch.cuda.nvtx.range_pop = custom_range_pop

def custom_is_fp16_supported(self):
    return True

def custom_is_triton_supported(self):
    return False

# in device other than cuda, the cuda.nvtx.range_push or range_pop will cause error
def custom_range_push(msg):
    pass
    #if hasattr(torch.cuda.nvtx, 'range_push'):
    #    return torch.cuda.nvtx.range_push(msg)

def custom_range_pop():
    pass
    #if hasattr(torch.cuda.nvtx, 'range_pop'):
    #    return torch.cuda.nvtx.range_pop()

