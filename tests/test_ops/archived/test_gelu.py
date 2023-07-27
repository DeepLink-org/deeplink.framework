# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torch.nn as nn

device = torch.device("dipu")

m = nn.GELU()
x1 = torch.randn(5, requires_grad = True)
y1 = m(x1)
y1.backward(torch.ones_like(y1))

x2 = x1.detach().to(device)
x2.requires_grad = True
y2 = m(x2)
y2.backward(torch.ones_like(y2))

assert torch.allclose(y1, y2.cpu(), atol=1e-3)
assert torch.allclose(x1.grad, x2.grad.cpu(), atol=1e-3)