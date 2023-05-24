import torch
import torch_dipu

x = torch.randn(3,4).cuda()

y = x.clone()

x.fill_(2)
y.fill_(2)

assert torch.allclose(x.cpu(), y.cpu())