# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNonzero(TestCase):
    def test_nonzero(self):
        input = torch.tensor([1, 1, 1, 0, 1])

        self.assertEqual(torch.nonzero(input), torch.nonzero(input.cuda()))

        torch.tensor(
            [
                [0.6, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, -0.4],
            ]
        )

        self.assertEqual(torch.nonzero(input), torch.nonzero(input.cuda()))

        input = torch.tensor([1, 1, 1, 0, 1])

        self.assertEqual(
            torch.nonzero(input, as_tuple=True),
            torch.nonzero(input.cuda(), as_tuple=True),
        )

        input = torch.tensor(
            [
                [0.6, 0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, -0.4],
            ]
        )

        self.assertEqual(
            torch.nonzero(input, as_tuple=True),
            torch.nonzero(input.cuda(), as_tuple=True),
        )

        self.assertEqual(
            torch.nonzero(torch.tensor(5), as_tuple=True),
            torch.nonzero(torch.tensor(5).cuda(), as_tuple=True),
        )


if __name__ == "__main__":
    run_tests()
