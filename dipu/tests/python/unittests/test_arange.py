# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, skipOn


class TestArange(TestCase):
    def test_arange(self):
        self.assertEqual(
            torch.arange(5.0, 20.0, 1.0),
            torch.arange(5.0, 20.0, 1.0, device="dipu").cpu(),
        )
        self.assertEqual(
            torch.arange(5.0, 20.0, 0.1),
            torch.arange(5.0, 20.0, 0.1, device="dipu").cpu(),
        )

    @skipOn("MLU", "camb impl has bug")
    def test_arange_bugprone(self):
        self.assertEqual(torch.arange(5), torch.arange(5, device="dipu").cpu())
        self.assertEqual(
            torch.arange(5, 20, 1), torch.arange(5, 20, 1, device="dipu").cpu()
        )
        self.assertEqual(torch.arange(5.0), torch.arange(5.0, device="dipu").cpu())


if __name__ == "__main__":
    run_tests()
