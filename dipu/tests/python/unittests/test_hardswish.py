# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestHardswish(TestCase):
    def setUp(self):
        self.x = torch.arange(-6.00001, 6.00001, 0.01)  # avoid the breakpoints
        self.x1 = self.x.cuda()

    def test_hardswish(self):
        self.x.requires_grad = True
        self.x1.requires_grad = True

        y = torch.nn.functional.hardswish(self.x, inplace=False)
        y1 = torch.nn.functional.hardswish(self.x1, inplace=False)
        self.assertTrue(torch.allclose(y, y1.cpu(), atol=1e-3, rtol=1e-3))

        y.backward(torch.ones_like(y))
        y1.backward(torch.ones_like(y1))
        self.assertTrue(
            torch.allclose(self.x.grad, self.x1.grad.cpu(), atol=1e-3, rtol=1e-3)
        )

    def test_hardswish_inplace(self):
        y = torch.nn.functional.hardswish(self.x, inplace=True)
        y1 = torch.nn.functional.hardswish(self.x1, inplace=True)
        self.assertTrue(torch.allclose(y, y1.cpu(), atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.x, self.x1.cpu(), atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    run_tests()
