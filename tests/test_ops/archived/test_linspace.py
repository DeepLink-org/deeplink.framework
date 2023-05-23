import torch
import torch_dipu

device_dipu = torch.device("dipu")

print(torch.linspace(3, 10, steps=5,device = "privateuseone"))
print(torch.linspace(3, 10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(3, 10, steps=5,device = "privateuseone").cpu(),torch.linspace(3, 10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(-10, 10, steps=5,device = "privateuseone").cpu(),torch.linspace(-10, 10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(start=-10, end=10, steps=5,device = "privateuseone").cpu(),torch.linspace(start=-10, end=10, steps=5,device = "cpu"))
assert torch.allclose(torch.linspace(start=-10, end=10, steps=1,device = "privateuseone").cpu(),torch.linspace(start=-10, end=10, steps=1,device = "cpu"))
