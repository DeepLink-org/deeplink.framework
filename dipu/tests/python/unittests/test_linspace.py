# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLinspace(TestCase):
    def test_linspace(self):
        from torch_dipu.dipu import diputype

        # print(torch.linspace(3, 10, steps=5,device = diputype))
        # print(torch.linspace(3, 10, steps=5,device = "cpu"))
        self.assertEqual(
            torch.linspace(3, 10, steps=5, device=diputype).cpu(),
            torch.linspace(3, 10, steps=5, device="cpu"),
        )
        self.assertEqual(
            torch.linspace(-10, 10, steps=5, device=diputype).cpu(),
            torch.linspace(-10, 10, steps=5, device="cpu"),
        )
        self.assertEqual(
            torch.linspace(start=-10, end=10, steps=5, device=diputype).cpu(),
            torch.linspace(start=-10, end=10, steps=5, device="cpu"),
        )
        self.assertEqual(
            torch.linspace(start=-10, end=10, steps=1, device=diputype).cpu(),
            torch.linspace(start=-10, end=10, steps=1, device="cpu"),
        )


if __name__ == "__main__":
    run_tests()
