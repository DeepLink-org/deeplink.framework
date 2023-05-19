# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

# torch.eq(input, scalar)
result = torch.eq(input, 3)
print(result)

# torch.eq(input, tensor)
result = torch.eq(input, torch.rand(10).to(device))
print(result)

# tensor.eq(tensor)
result = input.eq(torch.rand(10).to(device))
print(result)

# tensor.eq(scalar)
result = input.eq(3)
print(result)

# tensor.eq_(tensor)
input.eq_(torch.rand(10).to(device))
print(input)

# tensor.eq_(scalar)
input = torch.rand(10).to(device)
input.eq_(3)
print(input)
