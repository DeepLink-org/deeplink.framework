# Copyright (c) 2024, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestZeros(TestCase):
    def test_zeros(self):
        device = torch.device("dipu")
        size = [5, 6]
        x = torch.zeros(size=size)
        y = torch.zeros(size=size, device=device)
        self.assertEqual(x, y.cpu(), exact_dtype=True)

    def test_zero_(self):
        size = [3, 5]
        y = torch.randn(size=size).cuda()
        x = y.cpu()
        x.zero_()
        y.zero_()
        self.assertEqual(x, y.cpu(), exact_dtype=True)


if __name__ == "__main__":
    run_tests()
