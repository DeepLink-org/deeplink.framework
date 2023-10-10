# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMaxPool2d(TestCase):
    def setUp(self):
        self.x = torch.randn((1, 3, 4, 4))

    def _test_max_pool2d(self, x: torch.Tensor, devicestr: str, return_indices: bool):
        device = torch.device(devicestr)
        x = x.to(device)
        x.requires_grad_(True)
        kernel_size = (2, 2)
        stride = (2, 2)
        if return_indices:
            out, indices = torch.nn.functional.max_pool2d(
                x, kernel_size=kernel_size, stride=stride, return_indices=True
            )
            # print(f"out = {out}\nindices = {indices}")
        else:
            out = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)(x)
            indices = None
            # print(f"out = {out}")

        grad_output = torch.ones_like(out).to(device)
        out.backward(grad_output)
        # print(f"x.grad = {x.grad}")

        return out, x.grad.clone(), indices

    def test_max_pool2d_with_indices(self):
        out1, grad1, ind1 = self._test_max_pool2d(self.x, "dipu", return_indices=True)
        out2, grad2, ind2 = self._test_max_pool2d(self.x, "cpu", return_indices=True)
        self.assertEqual(out1, out2)
        self.assertEqual(grad1, grad2)
        self.assertEqual(ind1, ind2)

    def test_max_pool2d(self):
        out1, grad1, _ = self._test_max_pool2d(self.x, "dipu", return_indices=False)
        out2, grad2, _ = self._test_max_pool2d(self.x, "cpu", return_indices=False)
        self.assertEqual(out1, out2)
        self.assertEqual(grad1, grad2)


if __name__ == "__main__":
    run_tests()
