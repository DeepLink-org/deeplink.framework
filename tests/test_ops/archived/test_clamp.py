import torch
import torch_dipu

a = torch.randn(4).cuda()
print(a)
print(torch.clamp(a, min=-0.5, max=0.5))
print(a.clamp_(min=0, max=0.3))

min = torch.linspace(-1, 1, steps=4).cuda()

print(a.clamp_(min=min)) # camb not support now
print(torch.clamp(a, min=min)) # camb not support now
