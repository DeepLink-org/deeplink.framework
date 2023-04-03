import torch
import torch_dipu

device = torch.device("dipu")
x = torch.randn(4, 1).to(device)
y = torch.randn(1, 4).to(device)
print(torch.mul(x, y))

x = x.cpu()
y = y.cpu()
print(torch.mul(x, y))

a = torch.randn(3).to(device)
print(f"a = {a}")
r = torch.mul(a, 100)
print(f"r = {r}")