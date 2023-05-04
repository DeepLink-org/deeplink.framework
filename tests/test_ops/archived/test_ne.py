import torch
import torch_dipu
device = torch.device("dipu")
input = torch.rand(10).to(device)

# torch.ne(input, scalar)
resune = torch.ne(input, 3)
print(resune)

# torch.ne(input, tensor)
resune = torch.ne(input, torch.rand(10).to(device))
print(resune)

# tensor.ne(tensor)
resune = input.ne(torch.rand(10).to(device))
print(resune)

# tensor.ne(scalar)
resune = input.ne(3)
print(resune)

# tensor.ne_(tensor)
input.ne_(torch.rand(10).to(device))
print(input)

# tensor.ne_(scalar)
input = torch.rand(10).to(device)
input.ne_(3)
print(input)
