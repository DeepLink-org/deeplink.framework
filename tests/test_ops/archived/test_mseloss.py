# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
# 创建预测值和目标值张量
predictions = torch.tensor([0.5, 0.8, 0.2]).to(device)
targets = torch.tensor([1.0, 0.7, 0.3]).to(device)

# 计算均方误差损失
loss = torch.nn.functional.mse_loss(predictions, targets,reduction='mean')
print(loss)
loss = torch.nn.functional.mse_loss(predictions, targets,reduction='sum')
# 打印损失值
print(loss)
loss = torch.nn.functional.mse_loss(predictions, targets,reduction='none')

print(loss)
