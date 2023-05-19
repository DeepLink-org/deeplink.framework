# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")

x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]).to(device)

sorted_x, sorted_indices = torch.sort(x,  dim=0, descending=False)
print(sorted_x)

sorted_x, sorted_indices = torch.sort(x, stable = True, dim=0, descending=True)
print(sorted_x)


x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]).to(device)
torch.sort(x, dim=0, out = (sorted_x, sorted_indices))
print(sorted_x)
