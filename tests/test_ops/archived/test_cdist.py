import torch
import torch_dipu


a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
y = torch.cdist(a, b, p=2)
y1 = torch.cdist(a.cuda(), b.cuda(), p=2)
assert torch.allclose(y, y1.cpu(), atol = 1e-3)

a = torch.randn(1, 32, 32)
b = torch.randn(32, 2, 48, 64, 32)
y = torch.cdist(a, b, p=1)
y1 = torch.cdist(a.cuda(), b.cuda(), p=1)
assert torch.allclose(y, y1.cpu(), atol = 1e-3)