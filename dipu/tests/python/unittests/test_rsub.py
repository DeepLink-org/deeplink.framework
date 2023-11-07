# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestRsub(TestCase):
    def _test_rsub(self, a: torch.Tensor, b: torch.Tensor, alpha=1):
        r1 = torch.rsub(a.cpu(), b.cpu(), alpha=alpha)
        r2 = torch.rsub(a.cuda(), b.cuda(), alpha=alpha).cpu()
        self.assertEqual(r1, r2)

    def _test_rsub_scalar(self, a: torch.Tensor, b: float, alpha=1.0):
        r1 = torch.rsub(a.cpu(), b, alpha=alpha)
        r2 = torch.rsub(a.cuda(), b, alpha=alpha).cpu()
        self.assertEqual(r1, r2)

    def test_rsub(self):
        a = torch.tensor((1, 2))
        b = torch.tensor((0, 1))
        self._test_rsub(a, b)
        self._test_rsub(torch.ones(4, 5), torch.ones(4, 5) * 10)
        self._test_rsub(torch.ones(4, 5) * 1.1, torch.ones(4, 5) * 5, alpha=4)

    def test_rsub_scalar(self):
        self._test_rsub_scalar(torch.ones(4, 5), 10)
        self._test_rsub_scalar(torch.ones(4, 5), 10, 2.5)


if __name__ == "__main__":
    run_tests()
