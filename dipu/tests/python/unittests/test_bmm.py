# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBmm(TestCase):
    def test_bmm(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        batch1 = torch.randn(3, 3, 2).to(dipu)
        batch2 = torch.randn(3, 2, 4).to(dipu)
        out_dipu = torch.bmm(batch1, batch2)
        out_cpu = torch.bmm(batch1.to(cpu), batch2.to(cpu))
        self.assertEqual(out_dipu.to(cpu), out_cpu)


if __name__ == "__main__":
    run_tests()
