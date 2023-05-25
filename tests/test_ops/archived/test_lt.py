# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

# torch.lt(input, scalar)
result = torch.lt(input, 3)
print(result)

# torch.lt(input, tensor)
result = torch.lt(input, torch.rand(10).to(device))
print(result)

# tensor.lt(tensor)
result = input.lt(torch.rand(10).to(device))
print(result)

# tensor.lt(scalar)
result = input.lt(3)
print(result)

# tensor.lt_(tensor)
input.lt_(torch.rand(10).to(device))
print(input)

# tensor.lt_(scalar)
input = torch.rand(10).to(device)
input.lt_(3)
print(input)
