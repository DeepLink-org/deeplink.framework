# Copyright (c) 2023, DeepLink.
import torch
import torch.nn.functional as F
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNllLoss(TestCase):
    @staticmethod
    def _run_nll_loss(input: torch.Tensor, target: torch.Tensor, devicestr: str):
        device = torch.device(devicestr)
        input = input.to(device)
        input.requires_grad_(True)
        target = target.to(device)
        loss = F.nll_loss(F.log_softmax(input, dim=1), target)
        # print(f"loss = {loss}")

        loss.backward()
        # print(f"input.grad = {input.grad}")

        return loss, input.grad.clone()

    def _test_nll_loss(self, input, target):
        loss1, grad1 = self._run_nll_loss(input, target, "dipu")
        loss2, grad2 = self._run_nll_loss(input, target, "cpu")
        self.assertEqual(loss1, loss2)
        self.assertEqual(grad1, grad2)

    def test_nll_loss(self):
        input = torch.randn(3, 5)
        target = torch.tensor([1, 0, 4])
        self._test_nll_loss(input, target)


if __name__ == "__main__":
    run_tests()
