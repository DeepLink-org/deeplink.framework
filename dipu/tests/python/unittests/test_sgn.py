# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSgn(TestCase):
    def test_sgn(self):
        x_cuda = torch.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j]).cuda()
        x_cpu = x_cuda.cpu()

        self.assertEqual(x_cuda.sgn().cpu(), x_cpu.sgn())


if __name__ == "__main__":
    run_tests()
