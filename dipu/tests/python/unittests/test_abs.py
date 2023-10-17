# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAbs(TestCase):
    def setUp(self):
        device = torch.device("dipu")
        self.input = torch.tensor([-1.2, 3.4, -5.6, 7.8]).to(device)
        self.output = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(device)
        self.expected_output = torch.tensor([1.2, 3.4, 5.6, 7.8]).to(device)

    def test_abs(self):
        self.output = torch.abs(self.input)
        self.assertEqual(self.output, self.expected_output)

    def test_abs_out(self):
        torch.abs(self.input, out=self.output)
        self.assertEqual(self.output, self.expected_output)

    def test_abs_(self):
        torch.abs_(self.input)
        self.assertEqual(self.input, self.expected_output)


if __name__ == "__main__":
    run_tests()
