# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestRemainder(TestCase):
    def _test_remainder(self, a: torch.Tensor, b: float):
        a = a.cuda()
        out = torch.remainder(a, b)
        gold = a - a.div(b, rounding_mode="floor") * b
        self.assertEqual(out, gold)

    def test_remainder(self):
        self._test_remainder(torch.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
        self._test_remainder(torch.tensor([1, 2, 3, 4, 5]), 1.5)


if __name__ == "__main__":
    run_tests()
