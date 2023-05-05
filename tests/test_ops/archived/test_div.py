import torch
import torch_dipu

device = torch.device("dipu")
x = torch.randn(4, 4).to(device)
y = torch.randn(1, 4).to(device)
print(x)
torch.div(x, y)
z = torch.div(x, y)
print(f"z = {z.cpu()}")

x = x.cpu()
y = y.cpu()
print(torch.div(x, y))

a = torch.randn(3).to(device)
r = torch.div(a, 0.5)
print(f"a = {a.cpu()}")
print(f"r = {r.cpu()}")
