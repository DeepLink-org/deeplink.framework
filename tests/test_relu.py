import torch
import torch_dipu

device = torch.device("dipu")

x = torch.randn(2, 3).to(device)

print(f"x = {x}")

print(f"x.relu() = {x.relu()}")

print(f"before relu_ x = {x}")
x.relu_()
print(f"after relu_ x = {x}")