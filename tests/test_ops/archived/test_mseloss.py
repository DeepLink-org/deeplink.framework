import torch
import torch_dipu
dipu = torch.device("dipu")
cpu = torch.device("cpu")

# 创建预测值和目标值张量
predictions = torch.tensor([0.5, 0.8, 0.2])
targets = torch.tensor([1.0, 0.7, 0.3])

# 计算均方误差损失
cpu = torch.nn.functional.mse_loss(predictions.to(cpu), targets.to(cpu),reduction='mean')
dipu = torch.nn.functional.mse_loss(predictions.to(dipu), targets.to(dipu),reduction='mean')
assert  torch.allclose(cpu, dipu.to(cpu))
cpu = torch.nn.functional.mse_loss(predictions.to(cpu), targets.to(cpu),reduction='sum')
dipu = torch.nn.functional.mse_loss(predictions.to(dipu), targets.to(dipu),reduction='sum')
assert  torch.allclose(cpu, dipu.to(cpu))
cpu = torch.nn.functional.mse_loss(predictions.to(cpu), targets.to(cpu),reduction='none')
dipu = torch.nn.functional.mse_loss(predictions.to(dipu), targets.to(dipu),reduction='none')
assert torch.allclose(cpu, dipu.to(cpu))
