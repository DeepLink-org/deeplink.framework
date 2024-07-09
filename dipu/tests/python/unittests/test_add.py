# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAdd(TestCase):
    def test_add(self):
        x = torch.randn(3, 4).cuda()
        y = x.cpu()

        self.assertEqual((x + x).cpu(), y + y)
        self.assertEqual((x + x).cpu(), y + y)

        x.add_(3)
        y.add_(3)
        self.assertEqual(x.cpu(), y)

        x.add_(3)
        y.add_(3)
        self.assertEqual(x.cpu(), y)

        x.add_(torch.ones_like(x))
        y.add_(torch.ones_like(y))
        self.assertEqual(x.cpu(), y)

    def test_add_self_is_scalar(self):
        x = torch.randn(3, 4).cuda()
        y = x.cpu()
        z_device = torch.add(2, x, alpha=0.3)
        z_cpu = torch.add(2, y, alpha=0.3)
        self.assertEqual(z_device.cpu(), z_cpu)


if __name__ == "__main__":
    run_tests()
