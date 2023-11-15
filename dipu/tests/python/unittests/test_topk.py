# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestTopk(TestCase):
    N = 10000
    K = 100

    def _test_topk(self, x: torch.Tensor, check_ind: bool, *args, **kwargs):
        device = torch.device("dipu")
        sx1, si1 = torch.topk(x.to(device), *args, **kwargs)
        sx2, si2 = torch.topk(x.cpu(), *args, **kwargs)
        sx3 = torch.empty_like(sx1)
        si3 = torch.empty_like(si1)
        torch.topk(x.to(device), out=(sx3, si3), *args, **kwargs)
        self.assertEqual(sx1, sx2)
        self.assertEqual(sx1, sx3)
        if check_ind:
            self.assertEqual(si1, si2)
            self.assertEqual(si1, si3)

    def test_topk(self):
        x = torch.randn(self.N)
        self._test_topk(x, False, k=self.K)
        x = torch.randperm(self.N)
        self._test_topk(x, True, k=self.K)


if __name__ == "__main__":
    run_tests()
