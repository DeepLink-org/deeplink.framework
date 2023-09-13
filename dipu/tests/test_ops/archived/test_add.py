import torch
import torch_dipu

x = torch.randn(3,4).cuda()
y = x.cpu()

assert torch.allclose((x + x).cpu(), y + y)
assert torch.allclose((x + x).cpu(), y + y)

x.add_(3)
y.add_(3)
assert torch.allclose(x.cpu(), y)

x.add_(3)
y.add_(3)
assert torch.allclose(x.cpu(), y)

x.add_(torch.ones_like(x))
y.add_(torch.ones_like(y))
assert torch.allclose(x.cpu(), y)