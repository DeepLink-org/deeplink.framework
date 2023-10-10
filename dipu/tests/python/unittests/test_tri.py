# Copyright (c) 2023, DeepLink.
from typing import Callable
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import dipu, TestCase, run_tests


class TestTri(TestCase):
    def _test_tri(self, fn: Callable):
        xa = torch.randn(3, 3).cuda()
        xb = xa.cpu()

        self.assertEqual(fn(xa).cpu(), fn(xb))
        self.assertEqual(fn(xa, diagonal=1), fn(xb, diagonal=1))
        self.assertEqual(fn(xa, diagonal=-1), fn(xb, diagonal=-1))

        ya = torch.randn(4, 6).cuda()
        yb = ya.cpu()
        self.assertEqual(fn(ya, diagonal=1), fn(yb, diagonal=1))
        self.assertEqual(fn(ya, diagonal=-1), fn(yb, diagonal=-1))

    def test_tril(self):
        self._test_tri(torch.tril)

    def test_triu(self):
        self._test_tri(torch.triu)


if __name__ == "__main__":
    run_tests()
