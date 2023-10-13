import torch_dipu
import torch
from torch.cuda.amp import autocast as autocast


# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")

# Autocast does not need to pass in torch.dtype, 
# in which case the default data type will be used.
# (We changed the default data type to fp16 in dipu/torch_dipu/dipu/amp.py)
with torch.autocast("cuda"):
    pass

with torch.autocast("cuda", torch.float16):
    c_float16 = torch.mm(a_float32, b_float32)
    with torch.autocast("cuda", enabled=False):
        c_float32 = torch.mm(a_float32, b_float32)

assert c_float16.dtype == torch.float16
assert c_float32.dtype == torch.float32

with autocast(dtype=torch.float16):
    d_float16 = torch.mm(a_float32, b_float32)
    with autocast(enabled=False):
        d_float32 = torch.mm(a_float32, b_float32)

assert d_float16.dtype == torch.float16
assert d_float32.dtype == torch.float32

if torch.cuda.is_bf16_supported():
    with torch.autocast("cuda", torch.bfloat16):
        c_bfloat16 = torch.mm(a_float32, b_float32)
    assert c_bfloat16.dtype == torch.bfloat16

    with autocast(dtype=torch.bfloat16):
        d_bfloat16 = torch.mm(a_float32, b_float32)
    assert d_bfloat16.dtype == torch.bfloat16
