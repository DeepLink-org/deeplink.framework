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
        x = x.cpu()

        y = y.cpu()
        expected_z = torch.div(x, y)

        self.assertTrue(torch.allclose(z.cpu(), expected_z, atol=1e-3, rtol=1e-3))

    def test_div_scalar(self):
        a = torch.randn(3).to(self.device)
        r = torch.div(a, 0.5)

        a = a.cpu()
        expected_r = torch.div(a, 0.5)

        self.assertTrue(torch.allclose(r.cpu(), expected_r, atol=1e-3, rtol=1e-3))

    def test_div_self_is_scalar(self):
        x = torch.randn(3, 4).cuda()
        y = x.cpu()
        for rounding_mode in ["trunc", "floor", None]:
            z_device = torch.div(2, x, rounding_mode=rounding_mode)
            z_cpu = torch.div(2, y, rounding_mode=rounding_mode)
            self.assertEqual(z_device, z_cpu)


if __name__ == "__main__":
    run_tests()
