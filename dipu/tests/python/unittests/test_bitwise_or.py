# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBitwiseOr(TestCase):
    def test_bitwise_or(self):
        a = torch.tensor([-1, -2, 3], dtype=torch.int8)
        b = torch.tensor([1, 0, 3], dtype=torch.int8)
        y1 = torch.bitwise_or(a, b)
        y2 = torch.bitwise_or(a.cuda(), b.cuda())
        self.assertEqual(y1, y2.cpu())

    def test_bitwise_or_bool(self):
        a = torch.tensor([True, True, False])
        b = torch.tensor([False, True, False])
        y1 = torch.bitwise_or(a, b)
        y2 = torch.bitwise_or(a.cuda(), b.cuda())
        self.assertEqual(y1, y2.cpu())


if __name__ == "__main__":
    run_tests()
