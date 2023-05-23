import torch
import torch_dipu

x = torch.randn(3,4).cuda()
y = x.cpu()

assert torch.allclose((x + x).cpu(), y)