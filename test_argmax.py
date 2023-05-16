import torch
import torch_dipu

device = torch.device("dipu")

a = torch.randn(4, 4).to(device)
print(a)
print(torch.argmax(a))

a = torch.randn(4, 4).to(device)
print(a)
print(torch.argmax(a, dim=1))