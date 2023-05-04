import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

# torch.ge(input, scalar)
result = torch.ge(input, 3)
print(result)

# torch.ge(input, tensor)
result = torch.ge(input, torch.rand(10).to(device))
print(result)

# tensor.ge(tensor)
result = input.ge(torch.rand(10).to(device))
print(result)

# tensor.ge(scalar)
result = input.ge(3)
print(result)

# tensor.ge_(tensor)
input.ge_(torch.rand(10).to(device))
print(input)

# tensor.ge_(scalar)
input = torch.rand(10).to(device)
input.ge_(3)
print(input)
