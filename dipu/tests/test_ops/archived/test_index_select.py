import torch
import torch_dipu
import torch.nn.functional as F

device = torch.device("dipu")

x = torch.randn(3, 4)
indices = torch.tensor([0, 2])
assert torch.allclose(torch.index_select(x, 0, indices),torch.index_select(x.to(device), 0, indices.to(device)).cpu())
assert torch.allclose(torch.index_select(x, 1, indices),torch.index_select(x.to(device), 1, indices.to(device)).cpu())
