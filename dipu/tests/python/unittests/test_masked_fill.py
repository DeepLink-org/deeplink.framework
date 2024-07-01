# Copyright (c) 2024, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMaskedSelect(TestCase):
    def test_masked_fill_scalar(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        input = torch.randn(1, 1, 4096, 4096)
        mask = torch.randn(4, 1, 4096, 4096)
        mask = mask.ge(0)
        cpu = torch.masked_fill(input.to(cpu), mask.to(cpu), 0)
        dipu = torch.masked_fill(input.to(dipu), mask.to(dipu), 0)
        self.assertEqual(cpu, dipu.to(cpu))

    def test_masked_fill_scalar_inp(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        input = torch.randn(4, 1, 4096, 4096)
        mask = torch.randn(1, 1, 4096, 4096)
        mask = mask.ge(0)
        input_cpu = input.to(cpu)
        input_dipu = input.to(dipu)
        input_cpu.masked_fill_(mask.to(cpu), 0)
        input_dipu.masked_fill_(mask.to(dipu), 0)
        self.assertEqual(input_cpu, input_dipu)

    def test_masked_fill(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        input = torch.randn(1, 1, 4096, 4096)
        mask = torch.randn(4, 1, 4096, 4096)
        value = torch.tensor(1)
        mask = mask.ge(0)
        input_cpu = input.to(cpu)
        input_dipu = input.to(dipu)
        cpu = input_cpu.masked_fill(mask.to(cpu), value.to(cpu))
        dipu = input_dipu.masked_fill(mask.to(dipu), value.to(dipu))
        self.assertEqual(cpu, dipu.to(cpu))


if __name__ == "__main__":
    run_tests()
