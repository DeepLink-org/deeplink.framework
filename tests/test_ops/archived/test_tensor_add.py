import torch
import torch_dipu

device = torch.device("dipu")
x = torch.randn(2, 2).to(device)
y = torch.randn(2).to(device)
print(x)
print(y)
print(x + y)

x = x.cpu()
y = y.cpu()
print(x + y)
