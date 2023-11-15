# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestIsnan(TestCase):
    def test_isnan(self):
        input_tensor = torch.tensor([1.0, float("nan"), 2.0, 4]).to("cuda")
        output_tensor = torch.isnan(input_tensor)
        expected_output = torch.tensor([False, True, False, False])

        self.assertEqual(output_tensor.cpu(), expected_output)


if __name__ == "__main__":
    run_tests()
