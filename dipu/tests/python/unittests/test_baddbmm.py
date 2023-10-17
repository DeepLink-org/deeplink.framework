# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBaddbmm(TestCase):
    def setUp(self):
        self.input = torch.randn(10, 3, 5).cuda()
        self.batch1 = torch.randn(10, 3, 4).cuda()
        self.batch2 = torch.randn(10, 4, 5).cuda()
        self.beta = 1
        self.alpha = 1

    def test_baddbmm(self):
        cpu_result = torch.baddbmm(
            self.input.cpu(),
            self.batch1.cpu(),
            self.batch2.cpu(),
            beta=self.beta,
            alpha=self.alpha,
        )
        device_result = torch.baddbmm(
            self.input.cuda(),
            self.batch1.cuda(),
            self.batch2.cuda(),
            beta=self.beta,
            alpha=self.alpha,
        )
        self.assertTrue(
            torch.allclose(cpu_result, device_result.cpu(), atol=1e-3, rtol=1e-3)
        )

    def test_baddbmm_out(self):
        cpu_result = self.input.cpu().baddbmm(
            self.batch1.cpu(), self.batch2.cpu(), beta=self.beta, alpha=self.alpha
        )
        device_result = self.input.cuda().baddbmm(
            self.batch1.cuda(), self.batch2.cuda(), beta=self.beta, alpha=self.alpha
        )
        self.assertTrue(
            torch.allclose(cpu_result, device_result.cpu(), atol=1e-3, rtol=1e-3)
        )

    def test_baddbmm_(self):
        cpu_result = self.input.cpu().baddbmm_(
            self.batch1.cpu(), self.batch2.cpu(), beta=self.beta, alpha=self.alpha
        )
        device_result = self.input.cuda().baddbmm_(
            self.batch1.cuda(), self.batch2.cuda(), beta=self.beta, alpha=self.alpha
        )
        self.assertTrue(
            torch.allclose(cpu_result, device_result.cpu(), atol=1e-3, rtol=1e-3)
        )


if __name__ == "__main__":
    run_tests()
