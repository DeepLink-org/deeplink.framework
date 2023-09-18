import torch
import torch_dipu

a = torch.tensor([-1, -2, 3], dtype=torch.int8)
b = torch.tensor([1, 0, 3], dtype=torch.int8)
y1 = torch.bitwise_or(a, b)
y2 = torch.bitwise_or(a.cuda(), b.cuda())
assert torch.allclose(y1, y2.cpu())

a = torch.tensor([True, True, False])
b = torch.tensor([False, True, False])
y1 = torch.bitwise_or(a, b)
y2 = torch.bitwise_or(a.cuda(), b.cuda())
assert torch.allclose(y1, y2.cpu())