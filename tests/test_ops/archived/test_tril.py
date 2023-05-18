import torch
import torch_dipu
device = torch.device("dipu")
tensor = torch.randn(3, 3).to(device)
out = torch.empty((3,3)).to(device)
# 使用 tril 函数将张量转换为下三角矩阵
lower_triangular = torch.tril(tensor, out = out)
lower_triangular = torch.tril(tensor)
# 打印生成的下三角矩阵
print(lower_triangular)