# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestTanh(TestCase):
    def test_tanh(self):
        x1 = torch.randn(4, 5, 6, 7)
        x2 = x1.cuda()
        x1.requires_grad = True
        x2.requires_grad = True

        y1 = torch.tanh(x1)
        y2 = torch.tanh(x2)

        self.assertEqual(y1, y2.cpu(), prec=1e-3)

        y1.backward(torch.ones_like(y1))
        y2.backward(torch.ones_like(y2))

        self.assertEqual(x1.grad, x2.grad.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
