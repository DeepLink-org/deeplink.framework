# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSqrt(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.input = torch.rand(10).to(device)
        self.gold = torch.tensor([x**0.5 for x in self.input]).to(device)

    def test_sqrt(self):
        x = torch.sqrt(self.input)
        self.assertEqual(x, self.gold)

    def test_sqrt_(self):
        self.input.sqrt_()
        self.assertEqual(self.input, self.gold)


if __name__ == "__main__":
    run_tests()
