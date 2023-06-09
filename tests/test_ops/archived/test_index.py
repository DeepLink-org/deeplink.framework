import torch
import torch_dipu

x = torch.randn(3, 4, 5, 6).cuda()
index = torch.arange(0, 3, 1).cuda()
torch.allclose(x[index, :, :, :].cpu(), x.cpu()[index.cpu(), :, :, :])
torch.allclose(x[index, index, :, :].cpu(), x.cpu()[index.cpu(), index.cpu(), :, :])
torch.allclose(x[index, index, index, :].cpu(), x.cpu()[index.cpu(), index.cpu(), index.cpu(), :])
torch.allclose(x[index, index, index, index].cpu(), x.cpu()[index.cpu(), index.cpu(), index.cpu(), index.cpu()])
