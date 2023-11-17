# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import numpy as np

from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestGroupNorm(TestCase):
    def _init_module(self, num_groups, num_channels, affine=False):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.cpu_module = torch.nn.GroupNorm(
            self.num_groups, self.num_channels, affine=affine
        )
        self.dipu_module = torch.nn.GroupNorm(
            self.num_groups, self.num_channels, affine=affine
        )
        self.dipu_module = self.dipu_module.cuda()

    def _run_group_norm(self):
        x1 = self.x.clone().cpu()
        x1.requires_grad = True
        y1 = self.cpu_module(x1)
        y1.backward(torch.ones_like(y1))

        x2 = self.x.clone().cuda()
        x2.requires_grad = True
        # print(self.dipu_module, self.dipu_module.weight is None)
        y2 = self.dipu_module(x2)
        y2.backward(torch.ones_like(y2))

        self.assertTrue(torch.allclose(y1, y2.cpu(), atol=1e-3, rtol=1e-4))
        # print("GroupNorm forward ok")
        self.assertTrue(torch.allclose(x1.grad, x2.grad.cpu(), atol=1e-3, rtol=1e-4))
        # print("GroupNorm input.grad ok")
        if self.cpu_module.affine == True and self.cpu_module.weight.grad is not None:
            self.assertTrue(
                torch.allclose(
                    self.cpu_module.weight.grad,
                    self.dipu_module.weight.grad.cpu(),
                    atol=1e-3,
                    rtol=1e-4,
                )
            )
            self.assertTrue(
                torch.allclose(
                    self.cpu_module.bias.grad,
                    self.dipu_module.bias.grad.cpu(),
                    atol=1e-3,
                    rtol=1e-4,
                )
            )
            # print('GroupNorm param.grad ok')

    def test_group_norm_affine(self):
        self.x = torch.randn(20, 6, 10, 10)

        self._init_module(3, 6, affine=True)
        self._run_group_norm()
        self._init_module(6, 6, affine=True)
        self._run_group_norm()
        self._init_module(1, 6, affine=True)
        self._run_group_norm()

    # def test_group_norm_no_affine(self):
    #     self.x = torch.randn(20, 6, 10, 10)

    #     self._init_module(3, 6, affine=False)
    #     self._run_group_norm()
    #     self._init_module(6, 6, affine=False)
    #     self._run_group_norm()
    #     self._init_module(1, 6, affine=False)
    #     self._run_group_norm()


if __name__ == "__main__":
    run_tests()
