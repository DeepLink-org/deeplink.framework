# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestUnique(TestCase):
    def _test_unique(
        self, input, sorted=True, return_inverse=False, return_counts=False, dim=None
    ):
        r1 = torch.unique(input.cpu(), sorted, return_inverse, return_counts, dim)
        r2 = torch.unique(input.cuda(), sorted, return_inverse, return_counts, dim)
        self.assertEqual(r1, r2)

    def test_unique(self):
        x = torch.randn(30, 40, 20, 30)
        self._test_unique(x, return_inverse=False)
        self._test_unique(x, return_inverse=True)
        self._test_unique(x, return_inverse=False, return_counts=True)
        self._test_unique(x, return_inverse=True, return_counts=True)
        self._test_unique(x, return_inverse=False, dim=0)
        self._test_unique(x, return_inverse=True, dim=0)
        self._test_unique(x, return_inverse=False, return_counts=True, dim=0)
        self._test_unique(x, return_inverse=True, return_counts=True, dim=0)
        self._test_unique(x, return_inverse=True, return_counts=True, dim=2)


if __name__ == "__main__":
    run_tests()
