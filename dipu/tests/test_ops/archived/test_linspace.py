import torch
import torch_dipu
from torch_dipu.dipu import diputype

device_dipu = torch.device("dipu")

print(torch.linspace(3, 10, steps=5,device = diputype))
print(torch.linspace(3, 10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(3, 10, steps=5,device = diputype).cpu(),torch.linspace(3, 10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(-10, 10, steps=5,device = diputype).cpu(),torch.linspace(-10, 10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(start=-10, end=10, steps=5,device = diputype).cpu(),torch.linspace(start=-10, end=10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(start=-10, end=10, steps=1,device = diputype).cpu(),torch.linspace(start=-10, end=10, steps=1,device = "cpu"))
