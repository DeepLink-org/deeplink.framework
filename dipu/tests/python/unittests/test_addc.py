# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAddc(TestCase):
    def setUp(self):
        self.device = torch.device("dipu")
        self.t = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        self.t1 = torch.tensor([[1.0], [2.0], [3.0]]).to(self.device)
        self.t2 = torch.tensor([1.0, 2.0, 3.0]).to(self.device)

    def test_addcmul(self):
        output = torch.addcmul(self.t, self.t1, self.t2, value=0.1)
        expected_output = torch.tensor(
            [[1.1, 1.2, 1.3], [1.2, 1.4, 1.6], [1.3, 1.6, 1.9]]
        ).to(self.device)
        self.assertEqual(output, expected_output)

    def test_addcmul_(self):
        output = self.t.addcmul_(self.t2, self.t2, value=0.1)
        expected_output = torch.tensor([1.1, 1.4, 1.9]).to(self.device)
        self.assertEqual(output, expected_output)
        self.assertEqual(self.t, expected_output)

    def test_addcdiv(self):
        output = torch.addcdiv(self.t, self.t1, self.t2, value=0.1)
        expected_output = torch.tensor(
            [
                [1.1, 1.05, 1.0 + 0.1 / 3.0],
                [1.2, 1.1, 1.0 + 0.2 / 3.0],
                [1.3, 1.15, 1.1],
            ]
        ).to(self.device)
        self.assertEqual(output, expected_output)

    def test_addcdiv_(self):
        output = self.t.addcdiv_(self.t2, self.t2, value=0.1)
        expected_output = torch.tensor([1.1, 1.1, 1.1]).to(self.device)
        self.assertEqual(output, expected_output)
        self.assertEqual(self.t, expected_output)


if __name__ == "__main__":
    run_tests()
