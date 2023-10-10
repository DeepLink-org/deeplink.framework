# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNorm(TestCase):
    DIPU = torch.device("dipu")
    CPU = torch.device("cpu")

    def test_norm(self):
        x = torch.randn(3, 3, 2).to(self.DIPU)
        y = torch.norm(x)
        z = torch.norm(x.cpu())
        self.assertEqual(y.to(self.CPU), z)

        y = torch.norm(x, 2)
        z = torch.norm(x.cpu(), 2)
        self.assertEqual(y.to(self.CPU), z)


if __name__ == "__main__":
    run_tests()
