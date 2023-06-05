# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device('dipu')
# 不带out版本
x1 = torch.randn(2, 3).to(device)
x2 = torch.randn(2, 3).to(device)
stacked_tensor = torch.stack([x1, x2], dim=0)
print(stacked_tensor.shape)  # 输出torch.Size([2, 2, 3])

# 带out版本
out = torch.empty(2, 2, 3).to(device)
torch.stack([x1, x2], dim=0, out=out)
print(out.shape)  # 输出torch.Size([2, 2, 3])

# dim为负数的版本
a = torch.stack([x1.cpu(), x2.cpu(), x1.cpu(), x2.cpu()], dim=-2)
b = torch.stack([x1.to(device), x2.to(device), x1.to(device), x2.to(device)], dim=-2)
print(a.shape) # 输出torch.Size([2, 2, 3])
print(b.shape) # 输出torch.Size([2, 2, 3])

xx = torch.randn([3])
yy = torch.randn([3])
c = torch.stack([xx.cpu(), yy.cpu(), xx.cpu(), yy.cpu()], dim=-1)
d = torch.stack([xx.to('dipu'), yy.to('dipu'), xx.to('dipu'), yy.to('dipu')], dim=-1)
print(c.shape) # 输出torch.Size([3, 4])
print(d.shape) # 输出torch.Size([3, 4])
