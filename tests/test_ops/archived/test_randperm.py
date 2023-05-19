# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
out_dipu = torch.empty(4).to(device)
print(torch.randperm(4, out = out_dipu, device=device))

out_cpu = torch.empty(4)
print(torch.randperm(4, out = out_cpu))