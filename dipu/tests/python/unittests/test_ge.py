# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestGe(TestCase):
    def setUp(self):
        self.device = torch.device("dipu")
        self.input = torch.rand(10).to(self.device)
        self.input2 = torch.rand(10).to(self.device)

    def test_torch_ge_input_scalar(self):
        """torch.ge(input, scalar)"""
        result = torch.ge(self.input, 3)
        expected = torch.ge(self.input.cpu(), 3)
        self.assertEqual(result, expected)

    def test_torch_ge_input_tensor(self):
        """torch.ge(input, tensor)"""
        result = torch.ge(self.input, self.input2)
        expected = torch.ge(self.input.cpu(), self.input2.cpu())
        self.assertEqual(result, expected)

    def test_tensor_ge_tensor(self):
        """tensor.ge(tensor)"""
        result = self.input.ge(self.input2)
        expected = self.input.cpu().ge(self.input2.cpu())
        self.assertEqual(result, expected)

    def test_tensor_ge_scalar(self):
        """tensor.ge(scalar)"""
        result = self.input.ge(3)
        expected = self.input.cpu().ge(3)
        self.assertEqual(result, expected)

    def test_tensor_ge__tensor(self):
        """tensor.ge_(tensor)"""
        result = self.input
        expected = result.cpu()
        result.ge_(self.input2)
        expected.ge_(self.input2.cpu())
        self.assertEqual(result, expected)

    def test_tensor_ge__scalar(self):
        """tensor.ge_(scalar)"""
        result = self.input
        expected = result.cpu()
        self.input.ge_(3)
        expected.ge_(3)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
