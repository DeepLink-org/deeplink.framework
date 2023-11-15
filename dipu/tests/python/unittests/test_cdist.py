# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import (
    TestCase,
    onlyOn,
    run_tests,
    skipOn,
)


class TestCdist(TestCase):
    def _test_cdist(self, plist):
        a = torch.randn((5, 3), requires_grad=True)
        b = torch.randn((2, 3, 3), requires_grad=True)

        a_cuda = a.detach().cuda()
        b_cuda = b.detach().cuda()
        a_cuda.requires_grad = True
        b_cuda.requires_grad = True

        for p in plist:
            y = torch.cdist(a, b, p=p)
            y1 = torch.cdist(a_cuda, b_cuda, p=p)
            y.backward(torch.ones_like(y))
            y1.backward(torch.ones_like(y1))
            self.assertEqual(y, y1.cpu(), prec=1e-3)
            self.assertEqual(a.grad, a_cuda.grad.cpu(), prec=1e-3)
            self.assertEqual(b.grad, b_cuda.grad.cpu(), prec=1e-3)

    @onlyOn("MLU")
    def test_cdest_mlu(self):
        plist = [1]
        self._test_cdist(plist)

    @skipOn("MLU", "Currently only 1-norm is supported by camb for the scatter op")
    def test_cdest(self):
        plist = [1, 2, 0.5, float("inf")]
        self._test_cdist(plist)


if __name__ == "__main__":
    run_tests()
