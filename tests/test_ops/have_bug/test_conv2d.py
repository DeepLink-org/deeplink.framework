# Copyright (c) 2023, DeepLink.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch_dipu

device = torch.device("dipu")
m = nn.Conv2d(2, 3, 3, stride=2).to(device)
m.weight = nn.Parameter(torch.ones_like(m.weight))
m.bias = nn.Parameter(torch.ones_like(m.bias))
input_dipu = torch.randn(2, 2, 5, 5).to(device)
print(f"input_dipu = {input_dipu}")
print(f"m.weight = {m.weight}")
output_dipu = m(input_dipu)
print(output_dipu)

m = nn.Conv2d(2, 3, 3, stride=2)
m.weight = nn.Parameter(torch.ones_like(m.weight))
m.bias = nn.Parameter(torch.ones_like(m.bias))
input_cpu = input_dipu.cpu()
print(f"input_cpu = {input_cpu}")
print(f"m.weight = {m.weight}")
output_cpu = m(input_cpu)
print(output_cpu)

rtol = 1e-5
atol = 1e-8
assert np.allclose(output_cpu.detach().numpy(), output_dipu.detach().cpu().numpy(), rtol, atol, True)
print("conv2d output compare successfully")