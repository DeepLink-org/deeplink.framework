# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestUnfold(TestCase):
    def _run_unfold(self, devicestr: str):
        x0 = torch.arange(0, 12).reshape((3, 4))
        x = x0.to(dtype=torch.float32, device=devicestr)
        x.requires_grad = True
        y = x.unfold(0, 2, 1)
        # print(y.size(), y.stride())
        self.assertEqual(y.size(), torch.Size([2, 4, 2]))
        self.assertEqual(y.stride(), (4, 1, 4))
        grad_raw = torch.ones_like(y)
        y.backward(grad_raw)
        # print(x.grad)
        return y, x.grad

    def test_unfold(self):
        y1, grad1 = self._run_unfold("dipu")
        y2, grad2 = self._run_unfold("cpu")
        self.assertEqual(y1, y2)
        self.assertEqual(grad1, grad2)


if __name__ == "__main__":
    run_tests()
