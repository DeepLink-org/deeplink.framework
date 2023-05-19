# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

# torch.gt(input, scalar)
result = torch.gt(input, 3)
print(result)

# torch.gt(input, tensor)
result = torch.gt(input, torch.rand(10).to(device))
print(result)

# tensor.gt(tensor)
result = input.gt(torch.rand(10).to(device))
print(result)

# tensor.gt(scalar)
result = input.gt(3)
print(result)

# tensor.gt_(tensor)
input.gt_(torch.rand(10).to(device))
print(input)

# tensor.gt_(scalar)
input = torch.rand(10).to(device)
input.gt_(3)
print(input)
