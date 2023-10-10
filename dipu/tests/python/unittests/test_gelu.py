# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torch.nn as nn
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestGelu(TestCase):
    def test_gelu(self):
        device = torch.device("dipu")

        m = nn.GELU()
        x1 = torch.randn(5, requires_grad=True)
        y1 = m(x1)
        y1.backward(torch.ones_like(y1))

        x2 = x1.detach().to(device)
        x2.requires_grad = True
        y2 = m(x2)
        y2.backward(torch.ones_like(y2))

        self.assertEqual(y1, y2.cpu(), prec=1e-3)
        self.assertEqual(x1.grad, x2.grad.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
