import torch
import torch_dipu

x = torch.arange(-6, 6, 0.01)
x1 = x.cuda()

x.requires_grad = True
x1.requires_grad = True

y = torch.nn.functional.hardswish(x, inplace=False)
y1 = torch.nn.functional.hardswish(x1, inplace=False)

assert torch.allclose(y, y1.cpu(), atol = 1e-3, rtol = 1e-3)

y.backward(torch.ones_like(y))
y1.backward(torch.ones_like(y1))


y = torch.nn.functional.hardswish(x.clone(), inplace=True)
y1 = torch.nn.functional.hardswish(x1.clone(), inplace=True)

assert torch.allclose(y, y1.cpu(), atol = 1e-3, rtol = 1e-3)