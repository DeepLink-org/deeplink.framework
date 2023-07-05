# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torch.nn as nn

from torch_dipu.testing._internal.common_utils import create_common_tensor, TestCase, run_tests


class TestConvTranspose2d(TestCase):

    def init_module(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.cpu_module = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.dipu_module = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        if self.cpu_module.weight is not None:
            self.dipu_module.weight = nn.Parameter(self.cpu_module.weight.clone())
        if self.cpu_module.bias is not None:
            self.dipu_module.bias = nn.Parameter(self.cpu_module.bias.clone())
        self.dipu_module = self.dipu_module.cuda()

    def run_conv_transpose2d(self):
        x1 = self.x.clone().cpu()
        x1.requires_grad = True
        y1 = self.cpu_module(x1)
        y1.backward(torch.ones_like(y1))

        x2 = self.x.clone().cuda()
        x2.requires_grad = True
        print(self.dipu_module, self.dipu_module.weight is None)
        y2 = self.dipu_module(x2)
        y2.backward(torch.ones_like(y2))

        self.assertTrue(torch.allclose(y1, y2.cpu(), atol=1e-3, rtol=1e-4))
        print('ConvTranspose2d forward ok')
        self.assertTrue(torch.allclose(x1.grad, x2.grad.cpu(), atol=1e-3, rtol=1e-4))
        print('ConvTranspose2d input.grad ok')
        if self.cpu_module.weight.grad is not None:
            self.assertTrue(torch.allclose(self.cpu_module.weight.grad, self.dipu_module.weight.grad.cpu(), atol=1e-3, rtol=1e-4))
        if self.cpu_module.bias is not None and self.cpu_module.bias.grad is not None:
            self.assertTrue(torch.allclose(self.cpu_module.bias.grad, self.dipu_module.bias.grad.cpu(), atol=1e-3, rtol=1e-4))
        print('ConvTranspose2d param.grad ok')

    def test_conv_transpose2d_equal_stride(self):
        self.x = torch.randn(20, 16, 50, 20)

        self.init_module(16, 33, 3, stride=2)
        self.run_conv_transpose2d()

    def test_conv_transpose2d_unequal_stride(self):
        self.x = torch.randn(20, 16, 50, 20)

        self.init_module(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        self.run_conv_transpose2d()


if __name__ == "__main__":
    run_tests()