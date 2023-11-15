# Copyright (c) 2023, DeepLink.
import torch
import torch.nn as nn
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBatchNorm(TestCase):
    def test_batch_norm(self):
        device = torch.device("dipu")
        input = torch.randn(2, 3, 4).to(device)
        t = input.view(3, -1)
        m = torch.mean(t, 1)
        v = torch.var(t, 1)

        result_cpu = torch.nn.functional.batch_norm(input.cpu(), m.cpu(), v.cpu())
        result_device = torch.nn.functional.batch_norm(input, m, v)
        self.assertEqual(result_cpu, result_device.cpu(), prec=1e-3)

        # With Learnable Parameters
        # Without Learnable Parameters
        m = nn.BatchNorm2d(100, affine=True).cuda()
        input = torch.randn(20, 100, 35, 45).cuda()
        m(input)
        self.assertTrue(torch.ne(m.running_mean.cpu(), 0.0).any())
        self.assertTrue(torch.ne(m.running_var.cpu(), 1.0).any())


if __name__ == "__main__":
    run_tests()
