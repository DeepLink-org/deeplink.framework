# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNormal(TestCase):
    def test_normal_(self):
        dipu = torch.device("dipu")
        x = torch.randn(10000).to(dipu)
        x_bak = x.clone()
        x.normal_(mean=0.0, std=1.0)
        self.assertNotEqual(x, x_bak)
        self.assertEqual(x.mean().item(), 0.0, prec=0.01)
        self.assertEqual(x.std().item(), 1.0, prec=0.01)


if __name__ == "__main__":
    run_tests()
