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
        device = torch.device("dipu")
        size = [3, 5]
        x = torch.randn(size=size)
        y = torch.randn(size=size, device=device)
        self.assertEqual(x.zero_(), y.zero_().cpu(), exact_dtype=True)

if __name__ == "__main__":
    run_tests()
