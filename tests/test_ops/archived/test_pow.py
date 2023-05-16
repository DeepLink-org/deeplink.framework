import torch
import torch_dipu

a = torch.arange(2., 9.).cuda()
print(a)
print(torch.pow(a, 2))
print(a.pow_(2))



a = torch.arange(1., 5.).cuda()
exp = torch.arange(1., 5.).cuda()
print(a)
print(exp)
print(torch.pow(a, exp))
print(a.pow_(exp))