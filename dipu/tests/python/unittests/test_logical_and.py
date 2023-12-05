# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLogicalAnd(TestCase):
    def test_logical_and_bool(self):
        a = torch.tensor([True, False, True], dtype=torch.bool)
        b = torch.tensor([True, False, False], dtype=torch.bool)
        self.assertEqual(
            torch.logical_and(a.cuda(), b.cuda()
                              ).cpu(), torch.logical_and(a, b)
        )

    def test_logical_and_non_bool(self):
        a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        out = torch.tensor([], dtype=torch.bool)
        self.assertEqual(
            torch.logical_and(a.cuda(), b.cuda()
                              ).cpu(), torch.logical_and(a, b)
        )
        self.assertEqual(
            torch.logical_and(a.double().cuda(), b.double().cuda()).cpu(),
            torch.logical_and(a.double(), b.double()),
        )
        self.assertEqual(
            torch.logical_and(a.double().cuda(), b.cuda()).cpu(),
            torch.logical_and(a.double(), b),
        )
        self.assertEqual(
            torch.logical_and(
                a.cuda(), b.cuda(), out=torch.empty(4, dtype=torch.bool).cuda()
            ).cpu(),
            torch.logical_and(a, b, out=torch.empty(4, dtype=torch.bool)),
        )
        self.assertEqual(
            torch.logical_and(a.cuda(), b.cuda(), out=out.cuda()
                              ).cpu(), torch.logical_and(a, b, out=out)
        )


if __name__ == "__main__":
    run_tests()
