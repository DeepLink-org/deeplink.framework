# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestGarther(TestCase):
    def test_gather(self):
        device = torch.device("dipu")
        t = torch.tensor([[1, 2], [3, 4]])
        t2 = torch.tensor([[0, 0], [1, 0]])
        self.assertEqual(
            torch.gather(t, 1, t2, sparse_grad=True),
            torch.gather(t.to(device), 1, t2.to(device)).cpu(),
        )


if __name__ == "__main__":
    run_tests()
