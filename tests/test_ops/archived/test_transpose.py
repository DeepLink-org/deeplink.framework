# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

def print_tensor(y):
    print(y, y.shape, y.stride())

x = torch.randn(2, 3)
print_tensor(x)


y = torch.transpose(x, 0, 1)
print_tensor(y)

y = torch.transpose(x.cuda(), 0, 1)
print_tensor(y)

y = x.clone()
y.transpose_(0, 1)
print_tensor(y)

#have probel on camb
y = x.clone().cuda()
y.transpose_(0, 1)
print_tensor(y)
