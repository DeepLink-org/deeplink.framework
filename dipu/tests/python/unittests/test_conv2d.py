# Copyright (c) 2023, DeepLink.

import torch
import torch.nn as nn
import numpy as np
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestConv2D(TestCase):
    def test_conv_2d(self):
        device = torch.device("dipu")
        m = nn.Conv2d(2, 3, 3, stride=2).to(device)
        m.weight = nn.Parameter(torch.ones_like(m.weight))
        m.bias = nn.Parameter(torch.ones_like(m.bias))
        input_dipu = torch.randn(2, 2, 5, 5).to(device)
        # print(f"input_dipu = {input_dipu}")
        # print(f"m.weight = {m.weight}")
        output_dipu = m(input_dipu)
        # print(output_dipu)

        m = nn.Conv2d(2, 3, 3, stride=2)
        m.weight = nn.Parameter(torch.ones_like(m.weight))
        m.bias = nn.Parameter(torch.ones_like(m.bias))
        input_cpu = input_dipu.cpu()
        # print(f"input_cpu = {input_cpu}")
        # print(f"m.weight = {m.weight}")
        output_cpu = m(input_cpu)
        # print(output_cpu)

        self.assertTrue(
            np.allclose(
                output_cpu.detach().numpy(),
                output_dipu.detach().cpu().numpy(),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
            )
        )
        # print("conv2d output compare successfully")

    def test_conv2d_nhwc(self):
        device = torch.device("dipu")

        m = nn.Conv2d(2, 3, 3).to(device=device, memory_format=torch.channels_last)
        self.assertTrue(m.weight.is_contiguous(memory_format=torch.channels_last))

        x = torch.rand(2, 2, 5, 5).to(device=device, memory_format=torch.channels_last)
        x.requires_grad_()
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))

        y = m(x)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))

        y.backward(torch.rand_like(y))
        self.assertTrue(x.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(m.weight.grad.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":
    run_tests()
