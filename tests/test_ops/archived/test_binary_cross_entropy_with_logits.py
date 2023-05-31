import torch
import torch_dipu
import torch.nn.functional as F
device = torch.device("dipu")

input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
loss1 = F.binary_cross_entropy_with_logits(input, target)
loss2 = F.binary_cross_entropy_with_logits(input.to(device), target.to(device))
loss1.backward()
loss2.backward()
assert torch.allclose(loss1,loss2.cpu())