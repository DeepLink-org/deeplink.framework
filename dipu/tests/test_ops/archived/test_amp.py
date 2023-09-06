import torch_dipu
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 确定 CUDA 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化 GradScaler
scaler = GradScaler()

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for batch in range(100):
        # 生成模拟数据
        inputs = torch.randn(32, 10).to(device)
        targets = torch.randn(32, 10).to(device)

        # 使用 autocast 进行前向传播，根据需要动态地选择数据类型
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        # 使用 GradScaler 缩放损失，执行反向传播
        scaler.scale(loss).backward()
        # 使用 GradScaler 完成优化器的步骤更新
        scaler.step(optimizer)
        # 更新 GradScaler 的缩放因子
        scaler.update()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed!")
