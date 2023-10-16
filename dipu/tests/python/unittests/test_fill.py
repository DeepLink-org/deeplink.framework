# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestFill(TestCase):
    def test_fill(self):
        x = torch.randn(3, 4).cuda()
        y = x.clone()
        x.fill_(2)
        y.fill_(2)
        self.assertEqual(x.cpu(), y.cpu())


if __name__ == "__main__":
    run_tests()
