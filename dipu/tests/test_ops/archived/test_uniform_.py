# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
tensor = torch.empty(3, 3).to(device)

# 使用 uniform_ 函数生成均匀分布的随机数，范围在 [0, 1) 之间
tensor.uniform_()

# 打印生成的随机数张量
print(tensor)
