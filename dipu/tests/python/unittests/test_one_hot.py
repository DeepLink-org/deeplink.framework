# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestOneHot(TestCase):
    DIPU = torch.device("dipu")
    CPU = torch.device("cpu")

    def test_one_hot(self):
        self.assertEqual(
            torch.nn.functional.one_hot(torch.arange(0, 5).to(self.DIPU) % 3).to(
                self.CPU
            ),
            torch.nn.functional.one_hot(torch.arange(0, 5).to(self.CPU) % 3),
        )
        self.assertEqual(
            torch.nn.functional.one_hot(
                torch.arange(0, 5).to(self.DIPU) % 3, num_classes=5
            ).to(self.CPU),
            torch.nn.functional.one_hot(
                torch.arange(0, 5).to(self.CPU) % 3, num_classes=5
            ),
        )
        self.assertEqual(
            torch.nn.functional.one_hot(
                torch.arange(0, 6).to(self.DIPU).view(3, 2) % 3
            ).to(self.CPU),
            torch.nn.functional.one_hot(torch.arange(0, 6).to(self.CPU).view(3, 2) % 3),
        )


if __name__ == "__main__":
    run_tests()
