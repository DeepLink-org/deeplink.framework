# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestDiv(TestCase):
    def setUp(self):
        self.device = torch.device("dipu")

    def test_div_tensor(self):
        x = torch.randn(4, 4).to(self.device)
        y = torch.randn(1, 4).to(self.device)
        torch.div(x, y)
        z = torch.div(x, y)
        # print(f"z = {z.cpu()}")

        x = x.cpu()
        y = y.cpu()
        # print(torch.div(x, y))
        expected_z = torch.div(x, y)

        self.assertEqual(z, expected_z)

    def test_div_scalar(self):
        a = torch.randn(3).to(self.device)
        r = torch.div(a, 0.5)
        # print(f"a = {a.cpu()}")
        # print(f"r = {r.cpu()}")

        a = a.cpu()
        expected_r = torch.div(a, 0.5)

        self.assertEqual(r, expected_r)


if __name__ == "__main__":
    run_tests()
