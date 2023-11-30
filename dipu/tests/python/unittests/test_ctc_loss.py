# Copyright (c) 2023, DeepLink.
import numpy as np
import torch
import torch_dipu
import torch.nn.functional as F
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestCtcLoss(TestCase):
    def setUp(self):
        self.log_probs = torch.randn(50, 16, 20).log_softmax(2).detach()
        self.log_probs_device = self.log_probs.cuda()
        self.log_probs.requires_grad_()
        self.log_probs_device.requires_grad_()
        self.targets = torch.randint(1, 20, (16, 30), dtype=torch.long)

    def _check_ctc_loss_tensor(self):
        self.assertTrue(
            torch.allclose(self.loss1.cpu(), self.loss, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(
                self.log_probs_device.grad.cpu(),
                self.log_probs.grad,
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_ctc_loss_tensor_mean(self):
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        self.loss = F.ctc_loss(
            self.log_probs, self.targets, input_lengths, target_lengths
        )
        self.loss.backward()
        self.loss1 = F.ctc_loss(
            self.log_probs_device,
            self.targets.cuda(),
            input_lengths.cuda(),
            target_lengths.cuda(),
        )
        self.loss1.backward()
        self._check_ctc_loss_tensor()

    def test_ctc_loss_tensor_none(self):
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        self.loss = F.ctc_loss(
            self.log_probs,
            self.targets,
            input_lengths,
            target_lengths,
            reduction="none",
        )
        self.loss.backward(torch.ones_like(self.loss))
        self.loss1 = F.ctc_loss(
            self.log_probs_device,
            self.targets.cuda(),
            input_lengths.cuda(),
            target_lengths.cuda(),
            reduction="none",
        )
        self.loss1.backward(torch.ones_like(self.loss1))
        self._check_ctc_loss_tensor()

    def test_ctc_loss_intlist_none(self):
        input_lengths = tuple(np.array([50] * 16).astype(np.int64))
        target_lengths = tuple(np.random.randint(10, 30, (16,)).astype(np.int64))
        self.loss = F.ctc_loss(
            self.log_probs, self.targets, input_lengths, target_lengths
        )
        self.loss.backward()
        self.loss1 = F.ctc_loss(
            self.log_probs_device, self.targets.cuda(), input_lengths, target_lengths
        )
        self.loss1.backward()
        self._check_ctc_loss_tensor()


if __name__ == "__main__":
    run_tests()
