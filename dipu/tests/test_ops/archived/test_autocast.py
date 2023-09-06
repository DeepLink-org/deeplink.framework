import torch
import torch_dipu


# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")

with torch.autocast("cuda", torch.float16):
    c_float16 = torch.mm(a_float32, b_float32)
    with torch.autocast("cuda", torch.bfloat16):
        c_bfloat16 = torch.mm(a_float32, b_float32)
        with torch.autocast("cuda", enabled=False):
            c_float32 = torch.mm(a_float32, b_float32)

assert c_float16.dtype == torch.float16
assert c_bfloat16.dtype == torch.bfloat16
assert c_float32.dtype == torch.float32

from torch.cuda.amp import autocast as autocast

with autocast(dtype=torch.float16):
    c_float16 = torch.mm(a_float32, b_float32)
    with autocast(dtype=torch.bfloat16):
        c_bfloat16 = torch.mm(a_float32, b_float32)
        with autocast(enabled=False):
            c_float32 = torch.mm(a_float32, b_float32)

assert c_float16.dtype == torch.float16
assert c_bfloat16.dtype == torch.bfloat16
assert c_float32.dtype == torch.float32


