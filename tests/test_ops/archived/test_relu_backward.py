# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
x = torch.randn(3, 3).to(device)
x.requires_grad_(True)
print(x)

y = x.relu()
y.backward(torch.ones_like(y))

print(x.grad)
