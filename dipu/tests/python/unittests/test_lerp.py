# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLerp(TestCase):
    def test_lerp(self):
        start = torch.arange(1.0, 5.0)
        # end = torch.empty(4).fill_(10)

        device = "dipu"
        start_dipu = torch.arange(1.0, 5.0)
        end_dipu = torch.empty(4).fill_(10)
        cpu_result1 = torch.lerp(start_dipu, end_dipu, 0.5)
        cpu_result2 = torch.lerp(start_dipu, end_dipu, torch.full_like(start, 0.5))
        dipu_result1 = torch.lerp(start_dipu.to(device), end_dipu.to(device), 0.5)
        dipu_result2 = torch.lerp(
            start_dipu.to(device),
            end_dipu.to(device),
            torch.full_like(start, 0.5).to(device),
        )

        self.assertEqual(dipu_result1, cpu_result1)
        self.assertEqual(dipu_result2, cpu_result2)


if __name__ == "__main__":
    run_tests()
