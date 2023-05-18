import torch
import torch_dipu
device = torch.device("dipu")
# 创建输入张量
input_tensor = torch.tensor([-1.0, 2.0, -3.0]).to(device)

# 使用leaky_relu函数
output_tensor = torch.nn.functional.leaky_relu(input_tensor, negative_slope=0.1)
# 打印输出结果
print(output_tensor)
