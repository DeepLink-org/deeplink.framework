# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLogicalNot(TestCase):
    def test_logical_not_bool(self):
        a = torch.tensor([True, False, True], dtype=torch.bool)
        self.assertEqual(
            torch.logical_not(a.cuda()).cpu(), torch.logical_not(a)
        )

    def test_logical_not_non_bool(self):
        a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        self.assertEqual(
            torch.logical_not(a.cuda()).cpu(), torch.logical_not(a)
        )
        self.assertEqual(
            torch.logical_not(a.double().cuda()).cpu(),
            torch.logical_not(a.double()),
        )


if __name__ == "__main__":
    run_tests()
