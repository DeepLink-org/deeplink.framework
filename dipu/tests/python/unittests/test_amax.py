# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAmax(TestCase):
    def test_amax(self):
        a = torch.randn(4, 4)
        y1 = torch.amax(a, 1)
        y2 = torch.amax(a.cuda(), 1)
        self.assertEqual(y1, y2.cpu())

        a = torch.randn(64, 1, 128)
        y1 = torch.amax(a, (1, 2))
        y2 = torch.amax(a.cuda(), (1, 2))
        self.assertEqual(y1, y2.cpu())

        a = torch.randn(128, 64, 3, 3)
        y1 = torch.amax(a, (-1, 2), True)
        y2 = torch.amax(a.cuda(), (-1, 2), True)
        self.assertEqual(y1, y2.cpu())


if __name__ == "__main__":
    run_tests()
