# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestStack(TestCase):
    DEVICE = torch.device("dipu")

    def setUp(self):
        self.x1 = torch.randn(2, 3).to(self.DEVICE)
        self.x2 = torch.randn(2, 3).to(self.DEVICE)

    def test_stack(self):
        stacked_tensor1 = torch.stack([self.x1.cpu(), self.x2.cpu()], dim=0)
        stacked_tensor2 = torch.stack([self.x1, self.x2], dim=0)
        self.assertEqual(stacked_tensor1, stacked_tensor2.cpu())

    def test_stack_out(self):
        out = torch.empty(2, 2, 3).to(self.DEVICE)
        y1 = torch.stack([self.x1, self.x2], dim=0, out=out)
        y2 = torch.stack([self.x1.cpu(), self.x2.cpu()], dim=0, out=out.cpu())
        self.assertEqual(y1.cpu(), y2)

    def test_stack_neg_dim(self):
        a = torch.stack(
            [self.x1.cpu(), self.x2.cpu(), self.x1.cpu(), self.x2.cpu()], dim=-2
        )
        b = torch.stack(
            [
                self.x1.to(self.DEVICE),
                self.x2.to(self.DEVICE),
                self.x1.to(self.DEVICE),
                self.x2.to(self.DEVICE),
            ],
            dim=-2,
        )
        self.assertEqual(a, b.cpu())

        xx = torch.randn([3])
        yy = torch.randn([3])
        c = torch.stack([xx.cpu(), yy.cpu(), xx.cpu(), yy.cpu()], dim=-1)
        d = torch.stack(
            [xx.to("dipu"), yy.to("dipu"), xx.to("dipu"), yy.to("dipu")], dim=-1
        )
        self.assertEqual(c, d.cpu())


if __name__ == "__main__":
    run_tests()
