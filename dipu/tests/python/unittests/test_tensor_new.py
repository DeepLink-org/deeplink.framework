# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestTensorNew(TestCase):
    def _test_tensor_new(self, devicestr: str):
        device = torch.device(devicestr)
        x = torch.randn(10, 10).to(device)
        y = torch.randn(2, 2).to(device)
        z = y.new(x.storage())
        self.assertEqual(x, z.reshape(10, 10))

    def test_tensor_new_cpu(self):
        self._test_tensor_new("cpu")

    def test_tensor_new_dipu(self):
        self._test_tensor_new("dipu")


if __name__ == "__main__":
    run_tests()
