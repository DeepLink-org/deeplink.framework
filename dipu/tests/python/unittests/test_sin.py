# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSin(TestCase):
    DIPU = torch.device("dipu")
    CPU = torch.device("cpu")

    def setUp(self):
        self.input_dipu = torch.Tensor([[1, 2, 4, 5, 0, -1]]).to(self.DIPU)
        self.input_cpu = torch.Tensor([[1, 2, 4, 5, 0, -1]]).to(self.CPU)

    def test_sin(self):
        out_dipu = torch.sin(self.input_dipu)
        out_cpu = torch.sin(self.input_cpu)
        self.assertEqual(out_dipu.to(self.CPU), out_cpu)

    def test_sin_out(self):
        out_dipu = torch.empty_like(self.input_dipu).to(self.DIPU)
        out_cpu = torch.empty_like(self.input_cpu).to(self.CPU)
        torch.sin(self.input_cpu, out=out_cpu)
        torch.sin(self.input_dipu, out=out_dipu)
        self.assertEqual(out_dipu.to(self.CPU), out_cpu)

    def test_sin_(self):
        self.input_dipu.sin_()
        self.input_cpu.sin_()
        self.assertEqual(self.input_dipu.to(self.CPU), self.input_cpu)


if __name__ == "__main__":
    run_tests()
