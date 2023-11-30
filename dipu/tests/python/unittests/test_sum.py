# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSum(TestCase):
    def test_sum(self):
        device = torch.device("dipu")
        x = torch.arange(4 * 5 * 6).view(4, 5, 6).to(device)
        y1 = torch.sum(x, (2, 1))
        x = x.cpu()
        y2 = torch.sum(x, (2, 1))
        self.assertEqual(y1.cpu(), y2)

    def test_logsumexp(self):
        # special test cases in the logsumexp op
        a = torch.randn(3, 3)
        y1 = torch.logsumexp(a, 1)
        y2 = torch.logsumexp(a.cuda(), 1)
        self.assertEqual(y1, y2.cpu())

    def test_logsumexp_dim(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y1 = torch.logsumexp(a, dim=1, keepdim=True)
        y2 = torch.logsumexp(a.cuda(), dim=1, keepdim=True)
        self.assertEqual(y1, y2.cpu())


if __name__ == "__main__":
    run_tests()
