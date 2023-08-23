import torch
import torch_dipu

x = torch.randn(3, 4, 5, 6).cuda()
index = torch.arange(0, 3, 1).cuda()
torch.allclose(x[index, :, :, :].cpu(), x.cpu()[index.cpu(), :, :, :])
torch.allclose(x[index, index, :, :].cpu(), x.cpu()[index.cpu(), index.cpu(), :, :])
torch.allclose(x[index, index, index, :].cpu(), x.cpu()[index.cpu(), index.cpu(), index.cpu(), :])
torch.allclose(x[index, index, index, index].cpu(), x.cpu()[index.cpu(), index.cpu(), index.cpu(), index.cpu()])

input = torch.arange(16).reshape(4,2,2).cuda()
index1 = torch.randint(3,(1,)).cuda()
index2 = torch.randint(2,(1,)).cuda()
index3 = torch.tensor([False, False]).cuda()
torch.allclose(input[index1].cpu(), input.cpu()[index1.cpu()])
torch.allclose(input[index1, index2].cpu(), input.cpu()[index1.cpu(), index2.cpu()])
torch.allclose(input[..., index2, ...].cpu(), input.cpu()[..., index2.cpu(), ...])

torch.allclose(input[index1, index2, index3].cpu(), input.cpu()[index1.cpu(), index2.cpu(), index3.cpu()])
torch.allclose(input[index1, index2, ...].cpu(), input.cpu()[index1.cpu(), index2.cpu(), ...])
torch.allclose(input[index1, ..., index3].cpu(), input.cpu()[index1.cpu(), ..., index3.cpu()])

# test empty index tensor
input = torch.arange(16).reshape(4,2,2).cuda()
idx = torch.tensor([], dtype=torch.long).reshape(0,3).cuda()
idx1 = torch.tensor([], dtype=torch.long).reshape(0,1).cuda()
torch.allclose(input[idx].cpu(), input.cpu()[idx.cpu()])
torch.allclose(input[idx, idx1].cpu(), input.cpu()[idx.cpu(), idx1.cpu()])

# test empty input tensor
input = torch.tensor([]).cuda()
idx = torch.tensor([], dtype=torch.bool).cuda()
torch.allclose(input[idx].cpu(), input.cpu()[idx.cpu()])