import torch
import torch_dipu
import torch.nn.functional as F
device = torch.device("dipu")

x = torch.arange(8).view(2, 2, 2)


print(torch.flip(x, [0, 1]))
print(torch.flip(x.to(device), [0, 1]))
assert torch.allclose(torch.flip(x, [0, 1]),torch.flip(x, [0, 1]))
