import torch
import torch_dipu
device = torch.device("dipu")

x = torch.tensor([1., 2.])
y1 = torch.sigmoid(x)
y2 = torch.sigmoid(x.to(device))
assert torch.allclose(y1,y2.cpu())