# Copyright (c) 2023, DeepLink.
import torch
import torch.nn.functional as F
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestHardtanh(TestCase):
    def _test_hardtanh(self, input, min_val=-1.0, max_val=1.0, inplace=False):
        cpu_input = input.cpu()
        device_input = input.cuda()
        if not inplace:
            device_input.requires_grad_(True)
            cpu_input.requires_grad_(True)
        cpu_output = F.hardtanh(cpu_input, min_val, max_val, inplace)
        device_output = F.hardtanh(device_input, min_val, max_val, inplace)
        self.assertEqual(device_output.cpu(), cpu_output)

        if not inplace:
            cpu_output.backward(torch.ones_like(cpu_output))
            device_output.backward(torch.ones_like(device_output))
            self.assertTrue(
                torch.allclose(
                    device_input.grad.cpu(), cpu_input.grad, atol=1e-2, rtol=1e-2
                )
            )

    def test_hardtanh(self):
        self._test_hardtanh(torch.randn(3, 5))
        self._test_hardtanh(torch.randn(34, 51))
        self._test_hardtanh(torch.randn(3, 5), min_val=-3.0)
        self._test_hardtanh(
            torch.randn(3, 5) * 20, min_val=-10, max_val=17.0, inplace=True
        )
        self._test_hardtanh(torch.randn(3, 5), max_val=3.0)

        self._test_hardtanh(
            torch.randn(3, 5) * 20, min_val=-10, max_val=17.0, inplace=False
        )  # camb has bug ,fallback now
        self._test_hardtanh(
            torch.randn(3, 5) * 10, min_val=-20, max_val=17.0
        )  # camb has bug ,fallback now


if __name__ == "__main__":
    run_tests()
