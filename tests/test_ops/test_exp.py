import torch
import math
import torch_dipu

device = torch.device("dipu")
x = torch.tensor([0, math.log(2.)]).to(device)
print(torch.exp(x))
print(torch.Tensor.exp_(x))