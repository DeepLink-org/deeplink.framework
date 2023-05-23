import torch_dipu
import torch

a = torch.randn(2,2,4).cuda()
a.requires_grad=True
m = torch.nn.Linear(4,4, bias=False).cuda()
b = m(a)
loss = b.mean()
loss.backward()
