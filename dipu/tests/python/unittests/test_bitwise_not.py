# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBitwiseNot(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.x = torch.tensor([1, 2, 3]).to(device)
        self.expected = torch.tensor([-2, -3, -4]).to(device)

    def test_bitwise_not(self):
        y = self.x.bitwise_not()
        self.assertEqual(y, self.expected)
        y = torch.bitwise_not(self.x)
        self.assertEqual(y, self.expected)

    def test_bitwise_not_(self):
        y = self.x.bitwise_not_()
        self.assertEqual(y, self.expected)
        self.assertEqual(self.x, self.expected)


if __name__ == "__main__":
    run_tests()
