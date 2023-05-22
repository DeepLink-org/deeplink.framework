# Copyright (c) 2023, DeepLink.
import torch
from torch.nn import functional as F

import torch_dipu


device = torch.device("dipu")

x = torch.randn(3, 2).to(device)
y = torch.ones(3, 2).to(device)
print(x)
print(torch.where(x > 0, 2.0, 0.0))
print(torch.where(x > 0, x, y))
x = torch.randn(2, 2, dtype=torch.double).to(device)
print(x)
print(torch.where(x > 0, x, 0.))