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


if __name__ == "__main__":
    run_tests()
