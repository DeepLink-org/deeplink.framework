# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestRepeat(TestCase):
    def test_repeat(self):
        x = torch.randn(3)
        y = x.clone().cuda()
        self.assertEqual(x.repeat(4, 2), y.repeat(4, 2))


if __name__ == "__main__":
    run_tests()
