# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAdd(TestCase):
    def test_add(self):
        x = torch.randn(3, 4).cuda()
        y = x.cpu()

        self.assertEqual((x - x).cpu(), y - y)

        x.sub_(3)
        y.sub_(3)
        self.assertEqual(x.cpu(), y)

        x1 = torch.randn(3, 4).cuda()
        x -= x1
        y -= x1.cpu()
        self.assertEqual(x.cpu(), y)

        x = x - torch.ones_like(x)
        y = y - torch.ones_like(y)
        self.assertEqual(x.cpu(), y)


if __name__ == "__main__":
    run_tests()
