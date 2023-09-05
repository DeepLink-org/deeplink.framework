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

    def test_foreach_add__scalar(self):
        weights_cpu = []
        weights_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
        scalar = torch.randn(1).item()
        torch._foreach_add_(weights_cpu, scalar)
        torch._foreach_add_(weights_dipu, scalar)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_add_scalar(self):
        weights_cpu = []
        weights_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
        scalar = torch.randn(1).item()
        result_cpu = torch._foreach_add(weights_cpu, scalar)
        result_dipu = torch._foreach_add(weights_dipu, scalar)
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

    def test_foreach_mul__scalarlist(self):
        weights_cpu = []
        weights_dipu = []
        scalars = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
            scalar = torch.randn(1).item()
            scalars.append(scalar)
        torch._foreach_mul_(weights_cpu, scalars)
        torch._foreach_mul_(weights_dipu, scalars)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_mul_scalar(self):
        weights_cpu = []
        weights_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
        scalar = torch.randn(1).item()
        result_cpu = torch._foreach_mul(weights_cpu, scalar)
        result_dipu = torch._foreach_mul(weights_dipu, scalar)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_mul_scalar_list(self):
        weights_cpu = []
        weights_dipu = []
        scalars = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
            scalar = torch.randn(1).item()
            scalars.append(scalar)
        result_cpu = torch._foreach_mul(weights_cpu, scalars)
        result_dipu = torch._foreach_mul(weights_dipu, scalars)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_mul(self):
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

        result_cpu = torch._foreach_mul(weights_cpu, grads_cpu)
        result_dipu = torch._foreach_mul(weights_dipu, grads_dipu)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_neg(self):
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

    def test_foreach_div_(self):
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

        torch._foreach_div_(weights_cpu, grads_cpu)
        torch._foreach_div_(weights_dipu, grads_dipu)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_div__scalar(self):
        weights_cpu = []
        weights_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
        scalar = torch.randn(1).item()
        torch._foreach_div_(weights_cpu, scalar)
        torch._foreach_div_(weights_dipu, scalar)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_div__scalarlist(self):
        weights_cpu = []
        weights_dipu = []
        scalars = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
            scalar = torch.randn(1).item()
            scalars.append(scalar)
        torch._foreach_div_(weights_cpu, scalars)
        torch._foreach_div_(weights_dipu, scalars)
        for i in range(len(weights_cpu)):
            assert torch.allclose(weights_cpu[i], weights_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_div_scalar(self):
        weights_cpu = []
        weights_dipu = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
        scalar = torch.randn(1).item()
        result_cpu = torch._foreach_div(weights_cpu, scalar)
        result_dipu = torch._foreach_div(weights_dipu, scalar)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_div_scalar_list(self):
        weights_cpu = []
        weights_dipu = []
        scalars = []
        for i in range(100):
            x = torch.randn(3, 5)
            weights_cpu.append(x)
            weights_dipu.append(x.cuda())
            scalar = torch.randn(1).item()
            scalars.append(scalar)
        result_cpu = torch._foreach_div(weights_cpu, scalars)
        result_dipu = torch._foreach_div(weights_dipu, scalars)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_div(self):
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

        result_cpu = torch._foreach_div(weights_cpu, grads_cpu)
        result_dipu = torch._foreach_div(weights_dipu, grads_dipu)
        for i in range(len(weights_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_addcmul__scalar_list(self):
        inputs_cpu = []
        inputs_dipu = []
        tensor1s_cpu = []
        tensor1s_dipu = []
        tensor2s_cpu = []
        tensor2s_dipu = []
        scalars = []
        for i in range(100):
            x = torch.randn(30, 59)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor1s_cpu.append(x)
            tensor1s_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor2s_cpu.append(x)
            tensor2s_dipu.append(x.cuda())

            value = torch.randn(1).item()
            scalars.append(value)
        torch._foreach_addcmul_(self = inputs_cpu, tensor1 = tensor1s_cpu, tensor2 = tensor2s_cpu, scalars = scalars)
        torch._foreach_addcmul_(self = inputs_dipu, tensor1 = tensor1s_dipu, tensor2 = tensor2s_dipu, scalars = scalars)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_addcmul__scalar(self):
        inputs_cpu = []
        inputs_dipu = []
        tensor1s_cpu = []
        tensor1s_dipu = []
        tensor2s_cpu = []
        tensor2s_dipu = []
        for i in range(100):
            x = torch.randn(30, 59)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor1s_cpu.append(x)
            tensor1s_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor2s_cpu.append(x)
            tensor2s_dipu.append(x.cuda())

        value = torch.randn(1).item()
        torch._foreach_addcmul_(self = inputs_cpu, tensor1 = tensor1s_cpu, tensor2 = tensor2s_cpu, value = value)
        torch._foreach_addcmul_(self = inputs_dipu, tensor1 = tensor1s_dipu, tensor2 = tensor2s_dipu, value = value)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_addcmul__tensor(self):
        inputs_cpu = []
        inputs_dipu = []
        tensor1s_cpu = []
        tensor1s_dipu = []
        tensor2s_cpu = []
        tensor2s_dipu = []
        tensors = torch.randn(100)
        for i in range(100):
            x = torch.randn(30, 59)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor1s_cpu.append(x)
            tensor1s_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor2s_cpu.append(x)
            tensor2s_dipu.append(x.cuda())

        torch._foreach_addcmul_(self = inputs_cpu, tensor1 = tensor1s_cpu, tensor2 = tensor2s_cpu, scalars = tensors)
        torch._foreach_addcmul_(self = inputs_dipu, tensor1 = tensor1s_dipu, tensor2 = tensor2s_dipu, scalars = tensors.cuda())
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_sqrt_(self):
        inputs_cpu = []
        inputs_dipu = []
        for i in range(100):
            x = torch.rand(3, 5) * 100
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

        torch._foreach_sqrt_(inputs_cpu)
        torch._foreach_sqrt_(inputs_dipu)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_sqrt(self):
        inputs_cpu = []
        inputs_dipu = []
        for i in range(100):
            x = torch.rand(3, 5) * 100
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

        result_cpu = torch._foreach_sqrt(inputs_cpu)
        result_dipu = torch._foreach_sqrt(inputs_dipu)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(result_cpu[i], result_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_addcdiv__scalar_list(self):
        inputs_cpu = []
        inputs_dipu = []
        tensor1s_cpu = []
        tensor1s_dipu = []
        tensor2s_cpu = []
        tensor2s_dipu = []
        scalars = []
        for i in range(100):
            x = torch.randn(30, 59)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor1s_cpu.append(x)
            tensor1s_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor2s_cpu.append(x)
            tensor2s_dipu.append(x.cuda())

            value = torch.randn(1).item()
            scalars.append(value)
        torch._foreach_addcdiv_(self = inputs_cpu, tensor1 = tensor1s_cpu, tensor2 = tensor2s_cpu, scalars = scalars)
        torch._foreach_addcdiv_(self = inputs_dipu, tensor1 = tensor1s_dipu, tensor2 = tensor2s_dipu, scalars = scalars)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_addcdiv__scalar(self):
        inputs_cpu = []
        inputs_dipu = []
        tensor1s_cpu = []
        tensor1s_dipu = []
        tensor2s_cpu = []
        tensor2s_dipu = []
        for i in range(100):
            x = torch.randn(30, 59)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor1s_cpu.append(x)
            tensor1s_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor2s_cpu.append(x)
            tensor2s_dipu.append(x.cuda())

        value = torch.randn(1).item()
        torch._foreach_addcdiv_(self = inputs_cpu, tensor1 = tensor1s_cpu, tensor2 = tensor2s_cpu, value = value)
        torch._foreach_addcdiv_(self = inputs_dipu, tensor1 = tensor1s_dipu, tensor2 = tensor2s_dipu, value = value)
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)

    def test_foreach_addcdiv__tensor(self):
        inputs_cpu = []
        inputs_dipu = []
        tensor1s_cpu = []
        tensor1s_dipu = []
        tensor2s_cpu = []
        tensor2s_dipu = []
        tensors = torch.randn(100)
        for i in range(100):
            x = torch.randn(30, 59)
            inputs_cpu.append(x)
            inputs_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor1s_cpu.append(x)
            tensor1s_dipu.append(x.cuda())

            x = torch.randn(30, 59)
            tensor2s_cpu.append(x)
            tensor2s_dipu.append(x.cuda())

        torch._foreach_addcdiv_(self = inputs_cpu, tensor1 = tensor1s_cpu, tensor2 = tensor2s_cpu, scalars = tensors)
        torch._foreach_addcdiv_(self = inputs_dipu, tensor1 = tensor1s_dipu, tensor2 = tensor2s_dipu, scalars = tensors.cuda())
        for i in range(len(inputs_cpu)):
            assert torch.allclose(inputs_cpu[i], inputs_dipu[i].cpu(), atol = 1e-3, rtol = 1e-3)



if __name__ == "__main__":
    run_tests()
