# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

x = torch.all(input)
print(x)

input.all()

input = torch.tensor([[0, 1, 2], [3, 4, 5]]).to(device)
print(torch.all(input, dim=0, keepdim = True))