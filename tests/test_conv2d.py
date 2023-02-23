
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

torch.ops.load_library("../build/libtorch_dipu.so")

m = nn.Conv2d(2, 3, 3, stride=2, bias=False).cuda()
m.weight = nn.Parameter(torch.ones_like(m.weight))
input_cuda = torch.randn(2, 2, 5, 5).cuda()
print(f"input_cuda = {input_cuda}")
print(f"m.weight = {m.weight}")
output_cuda = m(input_cuda)
print(output_cuda)

m = nn.Conv2d(2, 3, 3, stride=2, bias=False)
m.weight = nn.Parameter(torch.ones_like(m.weight))
input_cpu = input_cuda.cpu()
print(f"input_cpu = {input_cpu}")
print(f"m.weight = {m.weight}")
output_cpu = m(input_cpu)
print(output_cpu)

rtol = 1e-5
atol = 1e-8
assert np.allclose(output_cpu.detach().numpy(), output_cuda.detach().cpu().numpy(), rtol, atol, True)
print("conv2d output compare successfully")