# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMseLoss(TestCase):
    def test_mse_loss(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")

        # 创建预测值和目标值张量
        predictions = torch.tensor([0.5, 0.8, 0.2])
        targets = torch.tensor([1.0, 0.7, 0.3])

        # 计算均方误差损失
        cpu = torch.nn.functional.mse_loss(
            predictions.to(cpu), targets.to(cpu), reduction="mean"
        )
        dipu = torch.nn.functional.mse_loss(
            predictions.to(dipu), targets.to(dipu), reduction="mean"
        )
        self.assertEqual(cpu, dipu.to(cpu))
        cpu = torch.nn.functional.mse_loss(
            predictions.to(cpu), targets.to(cpu), reduction="sum"
        )
        dipu = torch.nn.functional.mse_loss(
            predictions.to(dipu), targets.to(dipu), reduction="sum"
        )
        self.assertEqual(cpu, dipu.to(cpu))
        cpu = torch.nn.functional.mse_loss(
            predictions.to(cpu), targets.to(cpu), reduction="none"
        )
        dipu = torch.nn.functional.mse_loss(
            predictions.to(dipu), targets.to(dipu), reduction="none"
        )
        self.assertEqual(cpu, dipu.to(cpu))

    def test_mse_loss_backward(self):
        loss = torch.nn.MSELoss()
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)
        output = loss(input, target)
        output.backward()

        input2 = input.detach().clone().cuda()
        input2.requires_grad = True
        target2 = target.clone().cuda()
        output2 = loss(input2, target2)
        output2.backward()

        self.assertEqual(input.grad, input2.grad.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
