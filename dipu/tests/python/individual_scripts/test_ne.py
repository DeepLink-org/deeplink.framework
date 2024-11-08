# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLogicalNe(TestCase):
    def setUp(self):
        self.device = torch.device("dipu")

    def test_ne_tensor(self):
        x = torch.randn(5, 5)
        o = torch.randn(5, 5)

        res_on_cpu = torch.ne(x, o)
        res_on_npu = torch.ne(x.to(self.device), o.to(self.device))
        self.assertEqual(res_on_cpu, res_on_npu.cpu())

        s = torch.randn(1)
        res_on_cpu = torch.ne(x, s)
        res_on_npu = torch.ne(x.to(self.device), s.to(self.device))
        self.assertEqual(res_on_cpu, res_on_npu.cpu())

        res_on_cpu = torch.ne(s, x)
        res_on_npu = torch.ne(s.to(self.device), x.to(self.device))
        self.assertEqual(res_on_cpu, res_on_npu.cpu())

    def test_ne_tensor_scalar(self):
        x = torch.randn(5, 5)
        s = torch.rand(1).item()

        res_on_cpu = torch.ne(x, s)
        res_on_npu = torch.ne(x.to(self.device), s)
        self.assertEqual(res_on_cpu, res_on_npu.cpu())

    def test_ne_tensor_scalar_like(self):
        x = torch.rand(1)
        s = torch.rand(1)

        res_on_cpu = torch.ne(s, x)
        res_on_npu = torch.ne(s.to(self.device), x.to(self.device))
        self.assertEqual(res_on_cpu, res_on_npu.cpu())

if __name__ == "__main__":
    run_tests()
