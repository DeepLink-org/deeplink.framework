# Copyright (c) 2023, DeepLink.
import torch
import math
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestExp(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.x = torch.tensor([0, math.log(2.0)]).to(device)
        self.exp = torch.tensor([1.0, 2.0])

    def test_exp(self):
        y = torch.exp(self.x)
        self.assertEqual(y.cpu(), self.exp)

    def test_exp_(self):
        y = torch.Tensor.exp_(self.x)
        self.assertEqual(y.cpu(), self.exp)
        self.assertEqual(self.x.cpu(), self.exp)


if __name__ == "__main__":
    run_tests()
