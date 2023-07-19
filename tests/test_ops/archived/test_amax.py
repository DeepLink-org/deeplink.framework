import torch
import torch_dipu

a = torch.randn(4, 4)
y1 = torch.amax(a, 1)
y2 = torch.amax(a.cuda(), 1)
assert torch.allclose(y1, y2.cpu())


a = torch.randn(64, 1, 128)
y1 = torch.amax(a, (1, 2))
y2 = torch.amax(a.cuda(), (1, 2))
assert torch.allclose(y1, y2.cpu())


a = torch.randn(128, 64, 3, 3)
y1 = torch.amax(a, (-1, 2), True)
y2 = torch.amax(a.cuda(), (-1, 2), True)
assert torch.allclose(y1, y2.cpu())