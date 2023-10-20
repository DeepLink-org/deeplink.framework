# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAmp(TestCase):
    def test_autocast(self):
        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")

        with torch.autocast("cuda", torch.float16):
            c_float16 = torch.mm(a_float32, b_float32)
            with torch.autocast("cuda", enabled=False):
                c_float32 = torch.mm(a_float32, b_float32)

        self.assertEqual(c_float16.dtype, torch.float16)
        self.assertEqual(c_float32.dtype, torch.float32)

        infp16 = torch.arange(0, 8, dtype=torch.float16, device = "cuda").reshape(2, 4)
        with torch.autocast("cuda"):
          # amp will convert fp16 to fp32
          res = nn.functional.softmax(infp16, dim=0)
          self.assertEqual(res.dtype, torch.float32)

        with autocast(dtype=torch.float16):
            d_float16 = torch.mm(a_float32, b_float32)
            with autocast(enabled=False):
                d_float32 = torch.mm(a_float32, b_float32)

        self.assertEqual(d_float16.dtype, torch.float16)
        self.assertEqual(d_float32.dtype, torch.float32)

        # Autocast does not need to pass in torch.dtype,
        # in which case the default data type will be used.
        # (We changed the default data type to fp16 in dipu/torch_dipu/dipu/amp.py)
        with torch.autocast("cuda"):
            pass

        if torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", torch.bfloat16):
                c_bfloat16 = torch.mm(a_float32, b_float32)
            self.assertEqual(c_bfloat16.dtype, torch.bfloat16)

            with autocast(dtype=torch.bfloat16):
                d_bfloat16 = torch.mm(a_float32, b_float32)
            self.assertEqual(d_bfloat16.dtype, torch.bfloat16)

    def test_gradscaler(self):
        """won't fail, only detecting errors"""
        # 确定 CUDA 可用
        device = torch.device("cuda")

        # 定义一个简单的模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 初始化 GradScaler
        scaler = GradScaler()

        # 训练模型
        NUM_EPOCHS = 5
        for epoch in range(NUM_EPOCHS):
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

            # print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

        # print("Training completed!")


if __name__ == "__main__":
    run_tests()
