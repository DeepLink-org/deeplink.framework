# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
print(torch.randperm(4, device=device))

out_device = torch.randperm(4).cuda()
print(torch.randperm(4, out = out_device))