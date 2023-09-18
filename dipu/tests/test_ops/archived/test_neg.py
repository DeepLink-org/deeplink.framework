# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

x = torch.neg(input)
print(x)
print(input)
input.neg_()
print(input)