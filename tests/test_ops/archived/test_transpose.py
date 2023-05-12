import torch
import torch_dipu

x = torch.randn(2, 3)

def print_tensor(x):
    print(y, y.shape, y.stride())

y = torch.transpose(x, 0, 1)
print_tensor(y)

y = torch.transpose(x.cuda(), 0, 1)
print_tensor(y)

y = x.clone()
y.transpose_(0, 1)
print_tensor(y)

#have probel on camb
#y = x.clone().cuda()
#y.transpose_(0, 1)
#print_tensor(y)
