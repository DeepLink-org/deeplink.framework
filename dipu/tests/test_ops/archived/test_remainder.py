import torch
import torch_dipu

a = torch.tensor([-3., -2, -1, 1, 2, 3])
y1 = torch.remainder(a, 2)
y2 = torch.remainder(a.cuda(), 2)
print(y1)
assert torch.allclose(y1, y2.cpu())

a = torch.tensor([1, 2, 3, 4, 5])
y1 = torch.remainder(a, -1.5)
y2 = torch.remainder(a.cuda(), -1.5)
print(y2)
assert torch.allclose(y1, y2.cpu())