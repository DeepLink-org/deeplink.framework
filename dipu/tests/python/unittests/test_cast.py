# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestCast(TestCase):
    def test_cast(self):
        x = torch.arange(200).reshape(4, 50).cuda()
        y = torch.arange(200).reshape(4, 50)

        self.assertEqual(x.cpu(), y)
        self.assertEqual(x.double().cpu(), y.double())
        self.assertEqual(x.int().cpu(), y.int())
        self.assertEqual(x.long().cpu(), y.long())


if __name__ == "__main__":
    run_tests()
