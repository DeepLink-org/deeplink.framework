import torch
import torch_dipu

device = torch.device("dipu")
x = torch.randn(4, 4).to(device)
y = torch.randn(1, 4).to(device)
torch.div(x, y)
print(torch.div(x, y))

x = x.cpu()
y = y.cpu()
print(torch.div(x, y))

a = torch.randn(3).to(device)
print(f"a = {a}")
r = torch.div(a, 0.5)
print(f"r = {r}")
