# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestRsqrt(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.x = torch.rand(10).to(device)
        self.gold = torch.tensor([a**-0.5 for a in self.x])

    def test_rsqrt(self):
        y = torch.rsqrt(self.x)
        self.assertEqual(y, self.gold)

    def test_rsqrt_(self):
        self.x.rsqrt_()
        self.assertEqual(self.x, self.gold)


if __name__ == "__main__":
    run_tests()
