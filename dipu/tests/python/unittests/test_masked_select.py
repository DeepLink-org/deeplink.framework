# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMaskedSelect(TestCase):
    def test_masked_select(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        input = torch.randn(3, 4)
        mask = input.ge(0.5)
        cpu = torch.masked_select(input.to(cpu), mask.to(cpu))
        dipu = torch.masked_select(input.to(dipu), mask.to(dipu))
        self.assertEqual(cpu, dipu.to(cpu))


if __name__ == "__main__":
    run_tests()
