import torch
import torch_dipu

x = torch.arange(200).reshape(4,50).cuda()

print(x)
print(x.double())
print(x.int())
print(x.long())