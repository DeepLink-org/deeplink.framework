# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestFloorDivide(TestCase):
    def test_floor_divide(self):
        device = torch.device("dipu")
        a = torch.tensor([4.0, 3.0]).to(device)
        b = torch.tensor([2.0, 2.0]).to(device)
        y1 = torch.floor_divide(a, b)
        self.assertEqual(y1, torch.tensor([2.0, 1.0]))
        y2 = torch.floor_divide(a, 1.4)
        self.assertEqual(y2, torch.tensor([2.0, 2.0]))
        y3 = torch.floor_divide(2, 1.4)
        self.assertEqual(y3, torch.tensor(1.0))


if __name__ == "__main__":
    run_tests()
