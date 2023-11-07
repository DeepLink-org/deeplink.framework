# Copyright (c) 2023, DeepLink.
import torch
from torch.nn import functional as F
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestWhere(TestCase):
    DIPU = torch.device("dipu")

    def test_where_1(self):
        x = torch.randn(3, 2)
        z1 = torch.where(x.to(self.DIPU) > 0, 2.0, 0.0)
        z2 = torch.where(x > 0, 2.0, 0.0)
        self.assertEqual(z1, z2)

    def test_where_2(self):
        x = torch.randn(3, 2)
        y = torch.ones(3, 1)
        z1 = torch.where(x.to(self.DIPU) > 0, x.to(self.DIPU), y.to(self.DIPU))
        z2 = torch.where(x > 0, x, y)
        self.assertEqual(z1, z2)

    def test_where_3(self):
        x = torch.randn(3, 1, 1)
        y = torch.ones(3, 2, 1)
        z = torch.zeros(3, 1, 2)

        z1 = torch.where(x.to(self.DIPU) > 0, y.to(self.DIPU), z.to(self.DIPU))
        z2 = torch.where(x > 0, y, z)
        self.assertEqual(z1, z2)


if __name__ == "__main__":
    run_tests()
