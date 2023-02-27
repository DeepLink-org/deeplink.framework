import torch

torch.ops.load_library("../build/libtorch_dipu.so")

x = torch.randn(2, 2).cuda()
y = torch.randn(2).cuda()
print(x)
print(y)
print(x + y)

x = x.cpu()
y = y.cpu()
print(x + y)
