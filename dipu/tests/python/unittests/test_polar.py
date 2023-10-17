# Copyright (c) 2023, DeepLink.
import numpy as np
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestPolar(TestCase):
    def test_polar(self):
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        self.assertEqual(
            torch.polar(abs, angle), torch.polar(abs.cuda(), angle.cuda()).cpu()
        )


if __name__ == "__main__":
    run_tests()
