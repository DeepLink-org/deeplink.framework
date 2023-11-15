# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSilu(TestCase):
    def test_silu(self):
        DIPU = torch.device("dipu")
        CPU = torch.device("cpu")
        a = torch.randn(2)
        a_cpu = a.to(CPU)
        a_dipu = a.to(DIPU)
        silu_cpu = torch.nn.SiLU()(a_cpu)
        silu_dipu = torch.nn.SiLU()(a_dipu)
        self.assertEqual(silu_cpu, silu_dipu.to(CPU))
        torch.nn.SiLU(inplace=True)(a_cpu)
        torch.nn.SiLU(inplace=True)(a_dipu)
        self.assertEqual(a_cpu, a_dipu.to(CPU))
        # print(a_cpu, a_dipu.to(CPU))

    @staticmethod
    def _dodtype_silu(dtype):
        device = "cuda"
        m = torch.nn.SiLU()
        input = torch.ones(2, dtype=dtype).to(device)
        output = m(input)
        # print(output)
        return output

    def test_silu_cast(self):
        res1 = self._dodtype_silu(torch.half)
        res2 = self._dodtype_silu(torch.float32)
        self.assertEqual(res1.to(torch.float32), res2, prec=1e-02)


if __name__ == "__main__":
    run_tests()
