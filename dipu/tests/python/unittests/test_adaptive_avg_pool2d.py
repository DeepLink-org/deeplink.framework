# Copyright (c) 2023, DeepLink.
import torch
import torch.nn.functional as F
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAdaptiveAvgPool2D(TestCase):
    def test_adaptive_avg_pool2d(self):
        device = torch.device("dipu")
        x = torch.randn(1, 3, 32, 32).to(device)
        y = F.adaptive_avg_pool2d(x, (2, 2))
        x = x.cpu()
        expected_y = F.adaptive_avg_pool2d(x, (2, 2))
        self.assertEqual(y.cpu(), expected_y)


if __name__ == "__main__":
    run_tests()
