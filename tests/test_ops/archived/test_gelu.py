import torch
import torch_dipu
import torch.nn as nn

device = torch.device("dipu")

m = nn.GELU(approximate='tanh')
input = torch.randn(5)
print(input)
output = m(input)
print(output)
input = input.to(device)
output = m(input)
print(output)