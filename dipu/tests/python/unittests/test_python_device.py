# Copyright (c) 2024, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestPythonDevice(TestCase):
    def test_cpu(self):
        a = torch.tensor([1, 2, 3])
        self.assertEqual(str(a.device), "cpu")
        self.assertEqual(repr(a.device), "device(type='cpu')")
        self.assertEqual(str(a), "tensor([1, 2, 3])")
        self.assertEqual(repr(a), "tensor([1, 2, 3])")

    def test_cuda(self):
        device_index = 0  # NOTE: maybe 0 is not available, fix me if this happens
        torch.cuda.set_device(device_index)
        a = torch.tensor([1, 2, 3]).cuda()
        self.assertEqual(str(a.device), f"cuda:{device_index}")
        self.assertEqual(repr(a.device), f"device(type='cuda', index={device_index})")
        self.assertEqual(str(a), f"tensor([1, 2, 3], device='cuda:{device_index}')")
        self.assertEqual(repr(a), f"tensor([1, 2, 3], device='cuda:{device_index}')")


if __name__ == "__main__":
    run_tests()
