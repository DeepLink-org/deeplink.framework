# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAll(TestCase):
    def test_all(self):
        device = torch.device("dipu")
        input = torch.rand(10).to(device)

        x = torch.all(input).item()
        self.assertEqual(x, True)

        x = input.all().item()
        self.assertEqual(x, True)

        input = torch.tensor([[0, 1, 2], [3, 4, 5]]).to(device)
        x = torch.all(input, dim=0, keepdim=True)
        expect = torch.tensor([[False, True, True]]).to(device)
        self.assertEqual(x, expect)


if __name__ == "__main__":
    run_tests()
