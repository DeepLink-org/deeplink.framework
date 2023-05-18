import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

# torch.le(input, scalar)
result = torch.le(input, 3)
print(result)

# torch.le(input, tensor)
result = torch.le(input, torch.rand(10).to(device))
print(result)

# tensor.le(tensor)
result = input.le(torch.rand(10).to(device))
print(result)

# tensor.le(scalar)
result = input.le(3)
print(result)

# tensor.le_(tensor)
input.le_(torch.rand(10).to(device))
print(input)

# tensor.le_(scalar)
input = torch.rand(10).to(device)
input.le_(3)
print(input)
