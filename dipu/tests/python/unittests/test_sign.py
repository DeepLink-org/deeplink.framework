# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSign(TestCase):
    def test_sign(self):
        a = torch.tensor([0.7, -1.2, 0.0, 2.3])
        y1 = torch.sign(a)
        y2 = torch.sign(a.cuda())
        self.assertEqual(y1, y2.cpu())


if __name__ == "__main__":
    run_tests()
