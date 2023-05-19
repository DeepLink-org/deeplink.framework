# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu



device = torch.device("dipu")

a = torch.tensor([4.0, 3.0]).to(device)
b = torch.tensor([2.0, 2.0]).to(device)
print(torch.floor_divide(a, b))
print(torch.floor_divide(a, 1.4))
print(torch.floor_divide(2, 1.4))