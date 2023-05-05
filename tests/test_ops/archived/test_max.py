import torch
import torch_dipu
import torch.nn as nn

device = torch.device("dipu")
x = torch.randn(4).to(device)

y=torch.max(x)
print(x.cpu())
print(y.cpu())
