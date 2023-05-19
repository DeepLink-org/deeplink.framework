# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")

x = torch.tensor([-1.2, 3.4, -5.6, 7.8]).to(device)
y = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(device)
y = torch.abs(x)
torch.abs(x, out = y)
print(y)

input = torch.tensor([-1.2, 3.4, -5.6, 7.8]).to(device) 
input.abs_()
print(input)