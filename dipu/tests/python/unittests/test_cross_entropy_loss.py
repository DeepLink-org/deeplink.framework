# Copyright (c) 2023, DeepLink.
import torch
import torch.nn.functional as F
import torch_dipu
from torch_dipu.testing._internal.common_utils import run_tests, TestCase, skipOn


class TestCrossEntropyLoss(TestCase):
    @staticmethod
    def _cross_entropy_loss_eval(input, target, devicestr: str):
        device = torch.device(devicestr)
        input = input.to(device)
        input.requires_grad_(True)
        target = target.to(device)
        loss = F.cross_entropy(input, target)
        # print(f"loss = {loss}")
        loss.backward()
        # print(f"input.grad = {input.grad}")
        return input, loss

    def _test_cross_entropy_loss(self, input, target):
        input1, loss1 = self._cross_entropy_loss_eval(input, target, "dipu")
        input2, loss2 = self._cross_entropy_loss_eval(input, target, "cpu")
        self.assertEqual(loss1, loss2)
        self.assertEqual(input1.grad, input2.grad)

    def test_cross_entropy_loss_class_indices(self):
        """target with class indices"""
        input = torch.randn(3, 5)
        target = torch.randint(5, (3,), dtype=torch.int64)
        self._test_cross_entropy_loss(input, target)

    @skipOn("MLU", "Probabilities for each class are not supported by cnnl")
    def test_cross_entropy_loss_class_probabilities(self):
        """target with class probabilities"""
        input = torch.randn(3, 5)
        target = torch.randn(3, 5).softmax(dim=1)
        self._test_cross_entropy_loss(input, target)


if __name__ == "__main__":
    run_tests()
