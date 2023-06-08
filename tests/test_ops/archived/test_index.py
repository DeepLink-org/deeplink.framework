import torch
import torch_dipu

x = torch.randn(3, 4, 5, 6).cuda()
index = torch.arange(0, 3, 1).cuda()
torch.allclose(x[index, :, :, :].cpu(), x.cpu()[index.cpu(), :, :, :])
