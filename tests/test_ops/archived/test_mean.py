# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
a = torch.randn(4, 4).to(device)
print(torch.mean(a, 1, True))

a = a.cpu()
print(torch.mean(a, 1, True))
