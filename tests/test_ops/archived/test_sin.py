import torch
import torch_dipu

device = torch.device('dipu')
input = torch.Tensor([[1, 2, 4, 5,0,-1]]).to(device)
torch.sin(input)

out = torch.empty(input.size()).to(device)
print(torch.sin(input, out=out))
print(out)
input.sin_()