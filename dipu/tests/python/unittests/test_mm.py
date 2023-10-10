# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMm(TestCase):
    def test_mm(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        r1 = torch.mm(mat1.to(dipu), mat2.to(dipu))
        r2 = torch.mm(mat1.to(cpu), mat2.to(cpu))
        self.assertEqual(r1.to(cpu), r2)


if __name__ == "__main__":
    run_tests()
