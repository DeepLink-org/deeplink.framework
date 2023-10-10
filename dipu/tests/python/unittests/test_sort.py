# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSort(TestCase):
    N = 10000

    def _test_sort(self, x: torch.Tensor, check_ind: str, *args, **kwargs):
        device = torch.device("dipu")
        sx1, si1 = torch.sort(x.to(device), *args, **kwargs)
        sx2, si2 = torch.sort(x.cpu(), *args, **kwargs)
        sx3 = torch.empty_like(sx1)
        si3 = torch.empty_like(si1)
        torch.sort(x.to(device), out=(sx3, si3), *args, **kwargs)
        self.assertEqual(sx1, sx2)
        self.assertEqual(sx1, sx3)
        if check_ind == "check":
            self.assertEqual(si1, si2)
            self.assertEqual(si1, si3)
        elif check_ind == "check_fail":
            self.assertNotEqual(si1, si2)

    def test_sort(self):
        x = torch.empty(self.N).random_()
        self._test_sort(x, "skip", descending=False)
        self._test_sort(x, "skip", descending=True)

    def test_sort_dim(self):
        x = torch.empty(10, self.N).random_()
        self._test_sort(x, "skip", dim=0)
        self._test_sort(x, "skip", dim=1)

    def test_sort_stable(self):
        x = torch.empty(self.N).random_(0, 1)
        self._test_sort(x, "check_fail")
        self._test_sort(x, "check", stable=True)


if __name__ == "__main__":
    run_tests()
