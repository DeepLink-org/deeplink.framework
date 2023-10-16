# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torch.nn.functional as F
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestBinaryCrossEntropyWithLogits(TestCase):
    def test_binary_cross_entropy_with_logits(self):
        device = torch.device("dipu")
        input = torch.randn(3, requires_grad=True)
        target = torch.empty(3).random_(2)
        loss1 = F.binary_cross_entropy_with_logits(input, target)
        loss2 = F.binary_cross_entropy_with_logits(input.to(device), target.to(device))
        loss1.backward()
        loss2.backward()
        self.assertEqual(loss1, loss2.cpu())


if __name__ == "__main__":
    run_tests()
