# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestGenerator(TestCase):
    def test_foreach_add_(self):
        weights_cpu = []
        grads_cpu = []
        weights_dipu = []
        grads_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            y = torch.randn(3, 5)
            weights_cpu.append(x)
            grads_cpu.append(y)
            weights_dipu.append(x.cuda())
            grads_dipu.append(y.cuda())

        torch._foreach_add_(weights_cpu, grads_cpu, alpha = 1e-1)
        torch._foreach_add_(weights_dipu, grads_dipu, alpha = 1e-1)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_add(self):
        weights_cpu = []
        grads_cpu = []
        weights_dipu = []
        grads_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            y = torch.randn(3, 5)
            weights_cpu.append(x)
            grads_cpu.append(y)
            weights_dipu.append(x.cuda())
            grads_dipu.append(y.cuda())

        result_cpu = torch._foreach_add(weights_cpu, grads_cpu, alpha = 1e-1)
        result_dipu = torch._foreach_add(weights_dipu, grads_dipu, alpha = 1e-1)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_mul_(self):
        weights_cpu = []
        grads_cpu = []
        weights_dipu = []
        grads_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            y = torch.randn(3, 5)
            weights_cpu.append(x)
            grads_cpu.append(y)
            weights_dipu.append(x.cuda())
            grads_dipu.append(y.cuda())

        torch._foreach_mul_(weights_cpu, grads_cpu)
        torch._foreach_mul_(weights_dipu, grads_dipu)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_mul__scalar(self):
        weights_cpu = []
        weights_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
        scalar = torch.randn(1).item()
        torch._foreach_mul_(weights_cpu, scalar)
        torch._foreach_mul_(weights_dipu, scalar)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_ne(self):
        inputs_cpu = []
        inputs_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

        result_cpu = torch._foreach_neg(inputs_cpu)
        result_dipu = torch._foreach_neg(inputs_dipu)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)



if __name__ == "__main__":
    run_tests()
