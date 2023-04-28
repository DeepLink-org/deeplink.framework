import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

x = torch.sqrt(input)
print(x)

input.sqrt_()
print(input)