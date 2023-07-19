# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
x = torch.arange(4 * 5 * 6).view(4, 5, 6).to(device)
y1 = torch.sum(x, (2, 1))
x = x.cpu()
y2 = torch.sum(x, (2, 1))
assert torch.allclose(y1.cpu(), y2)

# special test cases in the logsumexp op
a = torch.randn(3, 3)
y1 = torch.logsumexp(a, 1)
y2 = torch.logsumexp(a.cuda(), 1)
assert torch.allclose(y1, y2.cpu())

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
y1 = torch.logsumexp(a, dim=1, keepdim=True)
y2 = torch.logsumexp(a.cuda(), dim=1, keepdim=True)
assert torch.allclose(y1, y2.cpu())
