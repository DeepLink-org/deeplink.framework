# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
# 创建一个示例张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).to(device)

# 在维度 1 上向右滚动
rolled = torch.roll(x, shifts=1, dims=1)

print(rolled)