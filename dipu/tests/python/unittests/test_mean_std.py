# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMeanStd(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.a = torch.randn(4, 4, 6, 7).to(device)

    def test_mean(self):
        self.assertTrue(
            torch.allclose(
                torch.mean(self.a, 1, True).cpu(),
                torch.mean(self.a.cpu(), 1, True),
                atol=1e-3,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.mean(self.a, (1, 3), True).cpu(),
                torch.mean(self.a.cpu(), (1, 3), True),
                atol=1e-3,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.mean(self.a, (1, 3), False).cpu(),
                torch.mean(self.a.cpu(), (1, 3), False),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_std(self):
        self.assertTrue(
            torch.allclose(
                torch.std(self.a, 1, True).cpu(),
                torch.std(self.a.cpu(), 1, True),
                atol=1e-3,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.std(self.a, (1, 3), True).cpu(),
                torch.std(self.a.cpu(), (1, 3), True),
                atol=1e-3,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.std(self.a, (1, 3), False).cpu(),
                torch.std(self.a.cpu(), (1, 3), False),
                atol=1e-3,
                rtol=1e-3,
            )
        )


if __name__ == "__main__":
    run_tests()
