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