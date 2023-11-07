# Copyright (c) 2023, DeepLink.
import math
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLog(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.input = torch.tensor([1.0, 2.0, 3.0]).to(device)

    def test_log(self):
        ans = torch.tensor([math.log(x) for x in self.input])
        y = torch.log(self.input)
        self.assertEqual(y, ans)

    def test_log_(self):
        ans = torch.tensor([math.log(x) for x in self.input])
        self.input.log_()
        self.assertEqual(self.input, ans)

    def test_log2(self):
        ans = torch.tensor([math.log2(x) for x in self.input])
        y = torch.log2(self.input)
        self.assertEqual(y, ans)

    def test_log2_(self):
        ans = torch.tensor([math.log2(x) for x in self.input])
        self.input.log2_()
        self.assertEqual(self.input, ans)


if __name__ == "__main__":
    run_tests()
