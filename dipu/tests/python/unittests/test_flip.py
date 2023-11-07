# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestFlip(TestCase):
    def test_flip(self):
        device = torch.device("dipu")
        x = torch.arange(8).view(2, 2, 2)
        y1 = torch.flip(x, [0, 1])
        y2 = torch.flip(x.to(device), [0, 1])
        self.assertEqual(y1, y2)


if __name__ == "__main__":
    run_tests()
