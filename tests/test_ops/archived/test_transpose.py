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


temp = torch.randn([16, 836, 32])
#temp = torch.load("transpose_error.pth")
a = temp.to('dipu').transpose(0, 1).contiguous()
b = temp.cpu().transpose(0, 1).contiguous()
ret1 = torch.allclose(b, a.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False)