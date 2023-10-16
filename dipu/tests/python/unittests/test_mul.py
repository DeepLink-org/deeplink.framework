# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMul(TestCase):
    def _test_mul(self, x, y):
        device = torch.device("dipu")
        z1 = torch.mul(x.to(device), y.to(device) if isinstance(y, torch.Tensor) else y)
        z2 = torch.mul(x.cpu(), y.cpu() if isinstance(y, torch.Tensor) else y)
        self.assertEqual(z1, z2)

    def test_mul_mm(self):
        """Vector/Matrix x Vector/Matrix"""
        x = torch.randn(4, 1)
        y = torch.randn(1, 4)
        self._test_mul(x, y)

    def test_mul_ms(self):
        """Vector/Matrix x Scalar"""
        a = torch.randn(3)
        self._test_mul(a, 100)


if __name__ == "__main__":
    run_tests()
