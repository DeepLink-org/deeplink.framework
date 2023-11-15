# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestReciprocal(TestCase):
    def test_reciprocal(self):
        DIPU = torch.device("dipu")
        CPU = torch.device("cpu")
        a = torch.randn(4)
        a_dipu = torch.reciprocal(a.to(DIPU))
        a_cpu = torch.reciprocal(a.to(CPU))
        self.assertEqual(a_cpu, a_dipu.to(CPU))
        torch.reciprocal(a.to(DIPU), out=a_dipu)
        torch.reciprocal(a.to(CPU), out=a_cpu)
        self.assertEqual(a_cpu, a_dipu.to(CPU))


if __name__ == "__main__":
    run_tests()
