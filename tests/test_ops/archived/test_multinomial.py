import torch
import torch_dipu
device = torch.device("dipu")
# 创建一个形状为 (1, 10) 的张量
probs = torch.tensor([0.1, 0.2, 0.3, 0.4]).to(device)
out = torch.empty((3,3),dtype=torch.long ).to(device)
# 使用 multinomial 函数从给定概率分布中生成样本
samples = torch.multinomial(probs, num_samples=5, replacement=True)
samples = torch.multinomial(probs, num_samples=5, replacement=True, out = out)
# 打印生成的样本
print(samples)
