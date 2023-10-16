# Copyright (c) 2023, DeepLink.
import math
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestUniform(TestCase):
    def _test_uniform_distribution(self, x: torch.Tensor, a, b):
        RTOL = 0.01
        expectation = (a + b) / 2
        std = math.sqrt((b - a) ** 2 / 12)
        self.assertEqual(x.mean().item(), expectation, prec=(b - a) * RTOL)
        self.assertEqual(x.std().item(), std, prec=std * RTOL)
        self.assertTrue(x.ge(a).all())
        self.assertTrue(x.less(b).all())
        self.assertEqual(x.min().item(), a, prec=(b - a) * RTOL)
        self.assertEqual(x.max().item(), b, prec=(b - a) * RTOL)

    def test_uniform_(self):
        N = 100
        DEVICE = torch.device("dipu")
        tensor = torch.empty(N, N).to(DEVICE)
        # 使用 uniform_ 函数生成均匀分布的随机数，范围在 [0, 1) 之间
        tensor.uniform_()
        self._test_uniform_distribution(tensor, 0, 1)


if __name__ == "__main__":
    run_tests()
