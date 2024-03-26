# Copyright (c) 2024, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestOnes(TestCase):
    def test_ones(self):
        device = torch.device("dipu")
        size = [5, 6]
        x = torch.ones(size=size)
        y = torch.ones(size=size, device=device)
        self.assertEqual(x, y.cpu(), exact_dtype=True)


if __name__ == "__main__":
    run_tests()
