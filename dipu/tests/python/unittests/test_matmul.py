# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, skipOn


class TestMatmul(TestCase):
    def _test_matmul(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        self.assertEqual(
            torch.matmul(tensor1, tensor2),
            torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(),
            prec=1e-3,
        )

    def test_matmul_vv(self):
        """vector x vector"""
        tensor1 = torch.randn(3)
        tensor2 = torch.randn(3)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_mv(self):
        """matrix x vector"""
        tensor1 = torch.randn(3, 4)
        tensor2 = torch.randn(4)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_bmbcv(self):
        """batched matrix x broadcasted vector"""
        tensor1 = torch.randn(10, 3, 4)
        tensor2 = torch.randn(4)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_bmbm(self):
        """batched matrix x batched matrix"""
        tensor1 = torch.randn(10, 3, 4)
        tensor2 = torch.randn(10, 4, 5)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_bmbcm1(self):
        """batched matrix x broadcasted matrix"""
        tensor1 = torch.randn(10, 3, 4)
        tensor2 = torch.randn(4, 5)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_bmbcm2(self):
        """batched matrix x broadcasted matrix"""
        tensor1 = torch.randn(20, 10, 3, 4)
        tensor2 = torch.randn(4, 5)
        self._test_matmul(tensor1, tensor2)

    @skipOn("MLU", "camb has problem")
    def test_matmul_bmbcm3(self):
        """batched matrix x broadcasted matrix"""
        tensor1 = torch.randn(20, 10, 3, 4)
        tensor2 = torch.randn(10, 4, 5)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_vm(self):
        """vector x matrix"""
        tensor1 = torch.randn(4)
        tensor2 = torch.randn(4, 10)
        self._test_matmul(tensor1, tensor2)

    def test_matmul_out(self):
        tensor1 = torch.randn(4, 20, 10)
        tensor2 = torch.randn(10, 50)
        out1 = torch.matmul(tensor1, tensor2)
        torch.matmul(tensor1, tensor2, out=out1)
        out2 = torch.ones_like(out1).cuda()
        torch.matmul(tensor1.cuda(), tensor2.cuda(), out=out2)
        self.assertEqual(out1, out2.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
