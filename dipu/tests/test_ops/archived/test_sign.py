import torch
import torch_dipu

a = torch.tensor([0.7, -1.2, 0., 2.3])
y1 = torch.sign(a)
y2 = torch.sign(a.cuda())
assert torch.allclose(y1, y2.cpu())