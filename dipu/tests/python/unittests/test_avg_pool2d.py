# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAvgPool2d(TestCase):
    def setUp(self):
        self.x = torch.randn((1, 3, 4, 4))

    def test_avg_pool2d(self):
        x = torch.randn((1, 3, 4, 4))
        kernel_size = (2, 2)
        stride = (2, 2)

        dipu_x = x.to("dipu").requires_grad_(True)
        dipu_y = torch.nn.functional.avg_pool2d(
            dipu_x, kernel_size=kernel_size, stride=stride
        )
        dipu_grady = torch.randn_like(dipu_y)
        dipu_y.backward(dipu_grady)

        cpu_x = x.requires_grad_(True)
        cpu_y = torch.nn.functional.avg_pool2d(
            cpu_x, kernel_size=kernel_size, stride=stride
        )
        cpu_grady = dipu_grady.cpu()
        cpu_y.backward(cpu_grady)

        self.assertEqual(dipu_y, cpu_y)
        self.assertEqual(dipu_x.grad, cpu_x.grad)


if __name__ == "__main__":
    run_tests()
