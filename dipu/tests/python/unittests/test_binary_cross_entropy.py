# Copyright (c) 2023, DeepLink.
from typing import Callable
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBinaryCrossEntropy(TestCase):
    def setUp(self):
        self.input = torch.randn(3, 2, requires_grad=True)
        self.target = torch.rand(3, 2, requires_grad=False)
        self.input1 = self.input.detach().clone().cuda()
        self.input1.requires_grad = True
        self.target1 = self.target.cuda()

    def _test_binary_cross_entropy(self, fn: Callable[..., torch.Tensor]):
        loss = fn(self.input, self.target)
        loss1 = fn(self.input1, self.target1)

        self.assertTrue(torch.allclose(loss1.cpu(), loss, atol=1e-3, rtol=1e-3))
        self.assertTrue(
            torch.allclose(
                self.input1.grad.cpu(), self.input.grad, atol=1e-3, rtol=1e-3
            )
        )

    def test_binary_cross_entropy(self):
        def fn(input, target):
            loss = torch.nn.functional.binary_cross_entropy(
                torch.sigmoid(input), target
            )
            loss.backward()
            return loss

        self._test_binary_cross_entropy(fn)

    def test_binary_cross_entropy_no_reduction(self):
        def fn(input, target):
            loss = torch.nn.functional.binary_cross_entropy(
                torch.sigmoid(input), target, reduction="none"
            )
            loss.backward(torch.ones_like(input))
            return loss

        self._test_binary_cross_entropy(fn)


if __name__ == "__main__":
    run_tests()
