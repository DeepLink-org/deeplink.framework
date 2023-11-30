# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSchema(TestCase):
    def test_floor(self):
        dipu = torch.device("dipu")
        cpu = torch.device("cpu")
        x = torch.randn(4)
        z1 = torch.floor(x)
        z2 = torch.floor(x.to(dipu))
        self.assertEqual(z1, z2.to(cpu))


if __name__ == "__main__":
    run_tests()
