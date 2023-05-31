# Copyright (c) 2023, DeepLink.
import torch
import torch.nn as nn
import torch_dipu

device = torch.device("dipu")
input = torch.randn(2, 3, 4).to(device)
t = input.view(3, -1)
m = torch.mean(t, 1)
v = torch.var(t, 1)

result_cpu = torch.nn.functional.batch_norm(input.cpu(), m.cpu(), v.cpu())
result_device = torch.nn.functional.batch_norm(input, m, v)
assert torch.allclose(result_cpu, result_device.cpu(), atol=1e-3)


# With Learnable Parameters
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=True).cuda()
input = torch.randn(20, 100, 35, 45).cuda()
output = m(input)
assert torch.ne(m.running_mean.cpu(), 0.0).any()
assert torch.ne(m.running_var.cpu(), 1.0).any()
