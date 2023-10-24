# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestRelu(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.x = torch.randn(20, 30).to(device)
        self.grad_gold = self.x.ge(0).to(self.x.dtype)
        self.y_gold = self.grad_gold.mul(self.x)

    def test_relu(self):
        y = self.x.relu()
        self.assertEqual(y, self.y_gold)

    def test_relu_(self):
        self.x.relu_()
        self.assertEqual(self.x, self.y_gold)

    def test_relu_backward(self):
        self.x.requires_grad_(True)
        y = self.x.relu()
        y.backward(torch.ones_like(y))

        self.assertEqual(y, self.y_gold)
        self.assertEqual(self.x.grad, self.grad_gold)


if __name__ == "__main__":
    run_tests()
