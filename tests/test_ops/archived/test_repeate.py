import torch
import torch_dipu

x = torch.tensor([1, 2, 3])
print(x, x.shape)
y = torch.tensor([1, 2, 3]).cuda()
print(x.repeat(4, 2).shape)
print(y.repeat(4, 2).shape)


