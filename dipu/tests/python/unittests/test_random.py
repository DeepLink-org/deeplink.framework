# Copyright (c) 2023, DeepLink.
import math
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, skipOn


class TestRandom(TestCase):
    def setUp(self):
        DEVICE = torch.device("dipu")
        N = 1000
        self.x = torch.empty(N, N).to(DEVICE)

    def _test_uniform_distribution(self, x: torch.Tensor, a, b):
        RTOL = 0.01
        expectation = (a + b) / 2
        std = math.sqrt((b - a) ** 2 / 12)
        self.assertEqual(x.mean().item(), expectation, prec=(b - a) * RTOL)
        self.assertEqual(x.std().item(), std, prec=std * RTOL)
        self.assertTrue(x.ge(a).all())
        self.assertTrue(x.le(b).all())
        self.assertEqual(x.min().item(), a, prec=(b - a) * RTOL)
        self.assertEqual(x.max().item(), b, prec=(b - a) * RTOL)

    def _test_dtype_default(self, dtype: torch.dtype, mantissa: int):
        x = self.x.to(dtype)
        y = x.random_()
        self.assertEqual(x, y)
        self._test_uniform_distribution(x, 0, 2**mantissa)

    # def test_random__fp16(self):
    #     self._test_dtype_default(torch.float16, 11)

    # def test_random__fp32(self):
    #     self._test_dtype_default(torch.float32, 24)

    # @skipOn("MLU", "camb does not support this type")
    # def test_random__fp64(self):
    #     self._test_dtype_default(torch.float64, 53)

    # def test_random__from_to(self):
    #     REP = 3
    #     for _ in range(REP):
    #         r = torch.empty(2).random_()
    #         a = int(r.min().item())
    #         b = int(r.max().item())
    #         y = self.x.random_(a, b + 1)
    #         self.assertEqual(self.x, y)
    #         self._test_uniform_distribution(self.x, a, b)


if __name__ == "__main__":
    run_tests()
