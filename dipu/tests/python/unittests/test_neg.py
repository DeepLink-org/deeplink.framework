# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNeg(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.input = torch.rand(10).to(device)
        self.ans = self.input.cpu().neg()

    def test_neg(self):
        x = torch.neg(self.input)
        self.assertEqual(x, self.ans)

    def test_neg_out(self):
        out_neg = torch.zeros_like(self.input).cuda()
        torch.neg(self.input, out=out_neg)
        self.assertEqual(out_neg, self.ans)

    def test_neg_(self):
        x = self.input.neg_()
        self.assertEqual(x, self.ans)
        self.assertEqual(self.input, self.ans)


if __name__ == "__main__":
    run_tests()
