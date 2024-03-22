# Copyright (c) 2023, DeepLink.
import torch
import math
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, onlyOn


class TestEqual(TestCase):
    @onlyOn("NPU")
    def test_equal_1(self):
        device = torch.device("dipu")
        x = torch.rand(3, 4).to(device)
        self.assertEqual(True, torch.equal(x, x))

    @onlyOn("NPU")
    def test_equal_2(self):
        device = torch.device("dipu")
        x = torch.rand(3, 4).to(device)
        self.assertEqual(False, torch.equal(x, x.to(torch.float16)))

    @onlyOn("NPU")
    def test_equal_3(self):
        device = torch.device("dipu")
        x = torch.zeros(3, 4).to(device)
        self.assertEqual(True, torch.equal(x, x.to(torch.float16)))

    @onlyOn("NPU")
    def test_equal_4(self):
        device = torch.device("dipu")
        x = torch.rand(3, 4).to(device)
        y = torch.rand(3, 5).to(device)
        self.assertEqual(False, torch.equal(x, y))


if __name__ == "__main__":
    run_tests()
