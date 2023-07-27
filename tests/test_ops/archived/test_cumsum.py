import torch
import torch_dipu


device = torch.device("dipu")
a = torch.randn(10)
y1 = torch.cumsum(a.to(device), dim=0)
y2 = torch.cumsum(a, dim=0)
assert torch.allclose(y1.cpu(), y2, atol=1e-3, rtol=1e-3)