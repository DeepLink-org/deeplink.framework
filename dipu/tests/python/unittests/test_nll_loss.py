# Copyright (c) 2023, DeepLink.
import torch
import torch.nn.functional as F
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNllLoss(TestCase):
    @staticmethod
    def _run_nll_loss(input: torch.Tensor, target: torch.Tensor, devicestr: str, **kwargs):
        device = torch.device(devicestr)
        input = input.to(device)
        input.requires_grad_(True)
        target = target.to(device)
        if 'weight' in kwargs:
            kwargs['weight'] = kwargs['weight'].to(device)
        loss = F.nll_loss(F.log_softmax(input, dim=1), target, **kwargs)

        # we do not use loss.backward because when reduction=='none'
        # loss is not a scalar
        grad_outputs = torch.ones_like(loss)
        grads = torch.autograd.grad(loss, input, grad_outputs=grad_outputs, create_graph=True)[0]

        return loss, grads.clone()

    def _test_nll_loss(self, input, target, **kwargs):
        loss1, grad1 = self._run_nll_loss(input, target, "dipu", **kwargs)
        loss2, grad2 = self._run_nll_loss(input, target, "cpu", **kwargs)
        self.assertEqual(loss1, loss2)
        self.assertEqual(grad1, grad2)

    def test_nll_loss(self):
        input = torch.randn(3, 5)
        target = torch.tensor([1, 0, 4])
        self._test_nll_loss(input, target)

    def test_nll_loss2d(self):
        input = torch.randn(1, 3, 2, 2)
        target = torch.tensor([[[0, 1], [2, 0]]])
        for reduction in ['none', 'mean', 'sum']:
            self._test_nll_loss(input, target, reduction=reduction)


if __name__ == "__main__":
    run_tests()
