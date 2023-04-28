import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

x = torch.any(input)
print(x)

input.any()

input = torch.tensor([[0, 1, 2], [3, 4, 5]]).to(device)
print(torch.any(input, dim=0, keepdim = True))