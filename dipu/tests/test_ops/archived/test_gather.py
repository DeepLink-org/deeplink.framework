import torch
import torch_dipu
import torch.nn.functional as F

device = torch.device("dipu")

t = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[0, 0], [1, 0]])
assert torch.allclose(torch.gather(t, 1, t2,sparse_grad=True),torch.gather(t.to(device), 1, t2.to(device)).cpu())

# 传入参数