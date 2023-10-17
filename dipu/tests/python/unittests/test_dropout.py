# Copyright (c) 2023, DeepLink.
from unittest import expectedFailure
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestDropout(TestCase):
    def setUp(self):
        self.input = torch.randn(4, 5)
        self.p = 0.8
        self.cpu_input = self.input.cpu().clone()
        self.device_input = self.input.cuda()
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

    @expectedFailure
    def test_dropout(self):
        """will fail since deterministic is not supported"""
        self.cpu_input.requires_grad_(True)
        cpu_out = torch.nn.functional.dropout(self.cpu_input, p=self.p)
        cpu_out.backward(torch.ones_like(self.cpu_input))
        # print('cpu_out:', self.cpu_out)
        # print('cpu_grad:', self.cpu_input.grad)
        self.device_input.requires_grad_(True)
        device_out = torch.nn.functional.dropout(self.device_input, p=self.p)
        device_out.backward(torch.ones_like(self.device_input))
        # print('device_out:', device_out)
        # print('device_grad:', device_input.grad)
        self.assertEqual(cpu_out, device_out)
        self.assertEqual(cpu_input.grad, device_input.grad)

    @expectedFailure
    def test_dropout_inplace(self):
        """will fail since deterministic is not supported"""
        cpu_out = torch.nn.functional.dropout(self.cpu_input, p=self.p, inplace=True)
        # print('cpu_out:', cpu_out)
        device_out = torch.nn.functional.dropout(
            self.device_input, p=self.p, inplace=True
        )
        # print('device_out:', device_out)
        self.assertEqual(cpu_out, device_out)


if __name__ == "__main__":
    run_tests()
