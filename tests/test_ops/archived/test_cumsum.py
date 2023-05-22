import torch
import torch_dipu


device = torch.device("dipu")
a = torch.randn(10)
print(a)
y1 = torch.cumsum(a.to(device), dim=0)
y2 = torch.cumsum(a, dim=0)
print(y1)
print(y2)
print(torch.allclose(y1.cpu(),y2))
assert torch.allclose(y1.cpu(),y2)