# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")

x1 = torch.randn(4, 5, 6, 7)
x2 = x1.cuda()
x1.requires_grad = True
x2.requires_grad = True

y1 = torch.tanh(x1)
y2 = torch.tanh(x2)

assert torch.allclose(y1, y2.cpu(), atol = 1e-3)

y1.backward(torch.ones_like(y1))
y2.backward(torch.ones_like(y2))

assert torch.allclose(x1.grad, x2.grad.cpu(), atol = 1e-3)