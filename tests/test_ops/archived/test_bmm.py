import torch
import torch_dipu

device = torch.device('dipu')

batch1 = torch.randn(3, 3, 2).to(device)
batch2 = torch.randn(3, 2, 4).to(device)

# 进行矩阵乘法
out = torch.bmm(batch1, batch2)

# 输出结果
print(out)
