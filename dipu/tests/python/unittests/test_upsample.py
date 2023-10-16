# Copyright (c) 2023, DeepLink.
import torch
import torch.nn as nn
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestUpsample(TestCase):
    def test_upsample(self):
        input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)

        m = nn.Upsample(scale_factor=2, mode="nearest")
        y1 = m(input)
        y2 = m(input.cuda())
        self.assertEqual(y1, y2.cpu(), prec=1e-3)

        m = nn.Upsample(scale_factor=2, mode="bilinear")  # align_corners=False
        y1 = m(input)
        y2 = m(input.cuda())
        self.assertEqual(y1, y2.cpu(), prec=1e-3)

        m = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        y1 = m(input)
        y2 = m(input.cuda())
        self.assertEqual(y1, y2.cpu(), prec=1e-3)

        # Try scaling the same data in a larger tensor
        input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
        input_3x3[:, :, :2, :2].copy_(input)
        # input_3x3

        m = nn.Upsample(scale_factor=2, mode="bilinear")  # align_corners=False
        # Notice that values in top left corner are the same with the small input (except at boundary)
        y1 = m(input_3x3)
        y2 = m(input_3x3.cuda())
        self.assertEqual(y1, y2.cpu(), prec=1e-3)

        m = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Notice that values in top left corner are now changed
        y1 = m(input_3x3)
        y2 = m(input_3x3.cuda())
        self.assertEqual(y1, y2.cpu(), prec=1e-3)

        m = nn.Upsample(scale_factor=2, mode="nearest")
        x1 = input.clone()
        x2 = input.cuda()
        x1.requires_grad = True
        x2.requires_grad = True
        y1 = m(x1)
        y2 = m(x2)
        y1.backward(torch.ones_like(y1))
        y2.backward(torch.ones_like(y2))
        self.assertEqual(y1, y2.cpu(), prec=1e-3)
        self.assertEqual(x1.grad, x2.grad.cpu(), prec=1e-3)

        m = nn.Upsample(scale_factor=2, mode="bilinear")  # align_corners=False
        x1 = input.clone()
        x2 = input.cuda()
        x1.requires_grad = True
        x2.requires_grad = True
        y1 = m(x1)
        y2 = m(x2)
        y1.backward(torch.ones_like(y1))
        y2.backward(torch.ones_like(y2))
        self.assertEqual(y1, y2.cpu(), prec=1e-3)
        self.assertEqual(x1.grad, x2.grad.cpu(), prec=1e-3)

        # torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        x1 = input.clone()
        x2 = input.cuda()
        x1.requires_grad = True
        x2.requires_grad = True
        y1 = torch.nn.functional.interpolate(
            x1, size=None, scale_factor=(4, 8), mode="nearest"
        )
        y2 = torch.nn.functional.interpolate(
            x2, size=None, scale_factor=(4, 8), mode="nearest"
        )
        y1.backward(torch.ones_like(y1))
        y2.backward(torch.ones_like(y2))
        self.assertEqual(y1, y2.cpu(), prec=1e-3)
        self.assertEqual(x1.grad, x2.grad.cpu(), prec=1e-3)

        x1 = input.clone()
        x2 = input.cuda()
        x1.requires_grad = True
        x2.requires_grad = True
        y1 = torch.nn.functional.interpolate(
            x1, size=None, scale_factor=(4, 8), mode="bilinear"
        )
        y2 = torch.nn.functional.interpolate(
            x2, size=None, scale_factor=(4, 8), mode="bilinear"
        )
        y1.backward(torch.ones_like(y1))
        y2.backward(torch.ones_like(y2))
        self.assertEqual(y1, y2.cpu(), prec=1e-3)
        self.assertEqual(x1.grad, x2.grad.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
