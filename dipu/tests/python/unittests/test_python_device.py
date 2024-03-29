# Copyright (c) 2024, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAbs(TestCase):
    def test_cpu(self):
        a = torch.tensor([1, 2, 3])
        self.assertEqual(str(a.device), "cpu")
        self.assertEqual(repr(a.device), "device(type='cpu')")
        self.assertEqual(str(a), "tensor([1, 2, 3])")
        self.assertEqual(repr(a), "tensor([1, 2, 3])")

    def test_cuda(self):
        torch.cuda.set_device(0)
        a = torch.tensor([1, 2, 3]).cuda()
        self.assertEqual(str(a.device), "cuda:0")
        self.assertEqual(repr(a.device), "device(type='cuda', index=0)")
        self.assertEqual(str(a), "tensor([1, 2, 3], device='cuda:0')")
        self.assertEqual(repr(a), "tensor([1, 2, 3], device='cuda:0')")


if __name__ == "__main__":
    run_tests()
