
import torch
import numpy as np
from torch.nn import functional as F
import torch_dipu

from torch_dipu.testing._internal.testcase import TestCase, run_tests
from torch_dipu.testing._internal.common_utils import cpu, dipu, get_dipu_device

class TestAdaptiveAvgPool2dBackward(TestCase):
    def cpu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        if input_x.dtype == torch.half:
            output = m(input_x.float()).half()
        else:
            output = m(input_x)
        output.backward(output)
        out = output.detach(), input_x.grad
        return out

    def dipu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        output = m(input_x)
        output.backward(output)
        out = output.detach().cpu(), input_x.grad.cpu()
        return out

    def test_adaptiveAvgPool2d_backward_1(self, device=get_dipu_device()):
        cpu_input = torch.randn((1, 64, 8, 9), dtype=torch.float32)
        dipu_input = cpu_input.to(device)
        output_size = np.array((2, 3))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        dipu_output = self.dipu_op_exec(dipu_input, output_size)
        self.assertRtolEqual(cpu_output[0], dipu_output[0])
        self.assertRtolEqual(cpu_output[1], dipu_output[1])

    def test_adaptiveAvgPool2d_backward_2(self, device=get_dipu_device()):
        cpu_input = torch.randn((1, 3, 3, 3), dtype=torch.float32)
        dipu_input = cpu_input.to(device)
        output_size = np.array((2, 2))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        dipu_output = self.dipu_op_exec(dipu_input, output_size)
        self.assertRtolEqual(cpu_output[0], dipu_output[0])
        self.assertRtolEqual(cpu_output[1], dipu_output[1])

    def test_adaptiveAvgPool2d_backward_fp16(self, device=get_dipu_device()):
        input_x = np.random.uniform(0, 1, (1, 3, 6, 6)).astype(np.float16)
        cpu_input = torch.from_numpy(input_x)
        dipu_input = cpu_input.to(device)
        output_size = np.array((5, 5))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        dipu_output = self.dipu_op_exec(dipu_input, output_size)
        self.assertRtolEqual(cpu_output[0], dipu_output[0])
        self.assertRtolEqual(cpu_output[1], dipu_output[1])


if __name__ == "__main__":
    run_tests()
