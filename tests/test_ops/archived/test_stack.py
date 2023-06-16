# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device('dipu')
# 不带out版本
x1 = torch.randn(2, 3).to(device)
x2 = torch.randn(2, 3).to(device)
stacked_tensor1 = torch.stack([x1.cpu(), x2.cpu()], dim=0)
stacked_tensor2 = torch.stack([x1, x2], dim=0)
assert torch.allclose(stacked_tensor1, stacked_tensor2.cpu())

# 带out版本
out = torch.empty(2, 2, 3).to(device)
y1 = torch.stack([x1, x2], dim=0, out=out)
y2 = torch.stack([x1.cpu(), x2.cpu()], dim=0, out=out.cpu())
assert torch.allclose(y1.cpu(), y2)

# dim为负数的版本
a = torch.stack([x1.cpu(), x2.cpu(), x1.cpu(), x2.cpu()], dim=-2)
b = torch.stack([x1.to(device), x2.to(device), x1.to(device), x2.to(device)], dim=-2)
assert torch.allclose(a, b.cpu())

xx = torch.randn([3])
yy = torch.randn([3])
c = torch.stack([xx.cpu(), yy.cpu(), xx.cpu(), yy.cpu()], dim=-1)
d = torch.stack([xx.to('dipu'), yy.to('dipu'), xx.to('dipu'), yy.to('dipu')], dim=-1)
assert torch.allclose(c, d.cpu())
