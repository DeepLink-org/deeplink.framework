# Copyright (c) 2023, DeepLink.
import torch
import torch.nn as nn
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLinear(TestCase):
    @staticmethod
    def _run_linear(x: torch.Tensor, label: torch.Tensor, devicestr: str):
        device = torch.device(devicestr)
        x = x.to(device)
        linear_layer = nn.Linear(3, 2).to(device)
        linear_layer.weight = nn.Parameter(torch.ones_like(linear_layer.weight))
        linear_layer.bias = nn.Parameter(torch.ones_like(linear_layer.bias))
        y_pred = linear_layer(x)
        # print(f"y_pred = \n{y_pred}")

        label = label.to(device)
        loss_fn = nn.MSELoss().to(device)
        loss = loss_fn(y_pred, label)
        loss.backward()
        # print(f"linear_layer.weight.grad = \n{linear_layer.weight.grad}")

        return y_pred.clone(), linear_layer.weight.grad.clone()

    def _test_linear(self, x: torch.Tensor, label: torch.Tensor):
        y_dipu, grad_dipu = self._run_linear(x, label, "dipu")
        y_cpu, grad_cpu = self._run_linear(x, label, "cpu")
        self.assertEqual(y_dipu, y_cpu)
        self.assertEqual(grad_dipu, grad_cpu, prec=1e-4)

    def test_linear_2d(self):
        x = torch.arange(9, dtype=torch.float).reshape(3, 3)
        label = torch.randn(3, 2)
        self._test_linear(x, label)

    def test_linear_3d(self):
        x = torch.arange(12, dtype=torch.float).reshape(2, 2, 3)
        label = torch.randn(2, 2, 2)
        self._test_linear(x, label)

    def test_linear_4d(self):
        x = torch.arange(24, dtype=torch.float).reshape(2, 2, 2, 3)
        label = torch.randn(2, 2, 2, 2)
        self._test_linear(x, label)

    @staticmethod
    def _run_linear_simple(a: torch.Tensor, m: torch.nn.Linear, devicestr: str):
        device = torch.device(devicestr)
        a = a.to(device).clone()
        a.requires_grad = True
        m = m.to(device)
        b = m(a)
        loss = b.mean()
        loss.backward()
        # print("a.grad:", a.grad)
        return b.clone(), a.grad.clone()

    def _test_linear_simple(self):
        x = torch.randn(2, 2, 4).cuda()
        m = torch.nn.Linear(4, 4, bias=False)
        y_dipu, grad_dipu = self._run_linear_simple(x, m, "dipu")
        y_cpu, grad_cpu = self._run_linear_simple(x, m, "cpu")
        self.assertEqual(y_dipu, y_cpu)
        self.assertEqual(grad_dipu, grad_cpu, prec=1e-4)

    def test_linear_simple(self):
        self._test_linear_simple()
        self._test_linear_simple()


if __name__ == "__main__":
    run_tests()
