# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestCumsum(TestCase):
    def test_cumsum(self):
        device = torch.device("dipu")
        a = torch.randn(10)
        y1 = torch.cumsum(a.to(device), dim=0)
        y2 = torch.cumsum(a, dim=0)
        self.assertEqual(y1.cpu(), y2)


if __name__ == "__main__":
    run_tests()
