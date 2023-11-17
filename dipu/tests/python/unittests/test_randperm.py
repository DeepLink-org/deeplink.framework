# Copyright (c) 2023, DeepLink.
import math
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, skipOn


class TestRandperm(TestCase):
    N = 100
    IOTA = torch.arange(0, N)

    def _test_is_perm(self, a: torch.Tensor):
        sorted_a, _ = a.sort()
        self.assertEqual(sorted_a, self.IOTA, exact_dtype=False)

    # def test_randperm(self):
    #     DEVICE = torch.device("dipu")
    #     a = torch.randperm(self.N, device=DEVICE)
    #     b = torch.randperm(self.N, device=DEVICE)
    #     self.assertNotEqual(a, b)
    #     self._test_is_perm(a)
    #     self._test_is_perm(b)

    def test_randperm_out(self):
        M = 1000
        out_device = torch.empty(M, self.N, dtype=torch.int64).cuda()
        for i in range(M):
            a = torch.randperm(self.N, out=out_device[i, :])
            self.assertEqual(a, out_device[i, :])
            self._test_is_perm(a)
        out_device = out_device.to(torch.float64)
        e = out_device.mean(dim=1)
        E = torch.ones_like(e).mul((self.N - 1) / 2)
        std = out_device.std(dim=1)
        STD = torch.ones_like(e).mul(math.sqrt((self.N - 1) ** 2 / 12))
        self.assertEqual(e, E, prec=1)
        self.assertEqual(std, STD, prec=1)

    @skipOn("MLU", "camb does not support this type")
    def test_randperm_out_fp32(self):
        out_device = torch.empty(self.N).cuda()
        a = torch.randperm(self.N, out=out_device)
        self.assertEqual(a, out_device)
        self._test_is_perm(out_device)


if __name__ == "__main__":
    run_tests()
