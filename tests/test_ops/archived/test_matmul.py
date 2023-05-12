import torch
import torch_dipu
device = torch.device("dipu")
# 创建两个矩阵
A = torch.tensor([[1.1, 2], [3, 4]]).to(device)
B = torch.tensor([[5, 6], [7., 8]]).to(device)

# 使用 matmul 函数进行矩阵乘法
C = torch.matmul(A, B).to(device)

# 打印结果
print(C)
D = torch.tensor([[1, 2], [3, 4]]).to(device)
torch.matmul(A, B, out = D)
print(D)