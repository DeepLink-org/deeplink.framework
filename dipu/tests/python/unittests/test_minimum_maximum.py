# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMinimimMaximum(TestCase):
    dipu = torch.device("dipu")
    cpu = torch.device("cpu")

    def test_minimum(self):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4))
        r_dipu = torch.minimum(a.to(self.dipu), b.to(self.dipu))
        r_cpu = torch.minimum(a.to(self.cpu), b.to(self.cpu))
        self.assertEqual(r_dipu.to(self.cpu), r_cpu)

    def test_minimum_scalar(self):
        # special test cases from the inference of internlm
        a = torch.randn((3, 4))
        b = torch.tensor(torch.finfo(a.dtype).max)
        # scalar on cpu
        r_dipu1 = torch.minimum(a.to(self.dipu), b)
        # scalar on device
        r_dipu2 = torch.minimum(a.to(self.dipu), b.to(self.dipu))
        r_cpu = torch.minimum(a, b)
        self.assertEqual(r_dipu1.to(self.cpu), r_cpu)
        self.assertEqual(r_dipu2.to(self.cpu), r_cpu)

    def test_minimum_different_devices(self):
        a = torch.tensor([1, -2, 3])
        b = torch.tensor([4, 0, 2]).to(self.dipu)
        with self.assertRaises(RuntimeError) as context:
            torch.minimum(a, b)
        self.assertIn(
            'Expected all tensors to be on the same device', str(context.exception))

    def test_maximum(self):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4))
        r_dipu = torch.maximum(a.to(self.dipu), b.to(self.dipu))
        r_cpu = torch.maximum(a.to(self.cpu), b.to(self.cpu))
        self.assertEqual(r_dipu.to(self.cpu), r_cpu)

    def test_maximum_scalar(self):
        # special test cases from the inference of internlm
        a = torch.randn((3, 4))
        b = torch.tensor(torch.finfo(a.dtype).min)
        # scalar on cpu
        r_dipu1 = torch.maximum(a.to(self.dipu), b)
        # scalar on device
        r_dipu2 = torch.maximum(a.to(self.dipu), b)
        r_cpu = torch.maximum(a, b)
        self.assertEqual(r_dipu1.to(self.cpu), r_cpu)
        self.assertEqual(r_dipu2.to(self.cpu), r_cpu)

    def test_maximum_different_devices(self):
        a = torch.tensor([1, -2, 3])
        b = torch.tensor([4, 0, 2]).to(self.dipu)
        with self.assertRaises(RuntimeError) as context:
            torch.maximum(a, b)
        self.assertIn(
            'Expected all tensors to be on the same device', str(context.exception))


if __name__ == "__main__":
    run_tests()
