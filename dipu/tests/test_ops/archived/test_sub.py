import torch
import torch_dipu

x = torch.randn(3,4).cuda()
y = x.cpu()

assert torch.allclose((x - x).cpu(), y - y)

x.sub_(3)
y.sub_(3)
assert torch.allclose(x.cpu(), y)

x1 = torch.randn(3,4).cuda()
x -= x1
y -= x1.cpu()
assert torch.allclose(x.cpu(), y)

x = x - torch.ones_like(x)
y = y - torch.ones_like(y)
assert torch.allclose(x.cpu(), y)