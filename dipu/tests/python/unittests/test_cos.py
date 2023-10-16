# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestCos(TestCase):
    def setUp(self):
        self.dipu = torch.device("dipu")
        self.cpu = torch.device("cpu")
        self.input_dipu = torch.Tensor([[1, 2, 4, 5, 0, -1]]).to(self.dipu)
        self.input_cpu = torch.Tensor([[1, 2, 4, 5, 0, -1]]).to(self.cpu)

    def test_cos(self):
        out_dipu = torch.cos(self.input_dipu)
        out_cpu = torch.cos(self.input_cpu)
        self.assertEqual(out_dipu.to(self.cpu), out_cpu)

    def test_cos_out(self):
        out_dipu = torch.empty_like(self.input_dipu).to(self.dipu)
        out_cpu = torch.empty_like(self.input_cpu).to(self.cpu)
        torch.cos(self.input_cpu, out=out_cpu)
        torch.cos(self.input_dipu, out=out_dipu)
        self.assertEqual(out_dipu.to(self.cpu), out_cpu)

    def test_cos_(self):
        self.input_dipu.cos_()
        self.input_cpu.cos_()
        self.assertEqual(self.input_dipu.to(self.cpu), self.input_cpu)


if __name__ == "__main__":
    run_tests()
