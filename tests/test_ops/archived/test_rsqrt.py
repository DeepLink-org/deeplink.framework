import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

x = torch.rsqrt(input)
print(x)

input.rsqrt_()
print(input)