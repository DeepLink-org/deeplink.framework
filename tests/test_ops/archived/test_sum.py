# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
x = torch.arange(4 * 5 * 6).view(4, 5, 6).to(device)
print(torch.sum(x, (2, 1)))

x = x.cpu()
print(torch.sum(x, (2, 1)))
