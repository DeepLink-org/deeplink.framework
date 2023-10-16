# Copyright (c) 2023, DeepLink.
import torch_dipu
import torch
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestAsstride(TestCase):
    @staticmethod
    def as_stride1(device):
        input1 = torch.arange(1, 301).to(device)
        dest1 = (
            torch.ones(30, 10)
            .to(device)
            .as_strided((30, 5), (5, 1), storage_offset=100)
        )
        ret1 = torch.as_strided(
            input1, dest1.size(), dest1.stride(), storage_offset=100
        )
        dest1.copy_(ret1)
        return dest1

    def test_as_stride1(self):
        dest_cpu = self.as_stride1("cpu")
        dest_cuda = self.as_stride1("cuda")
        ret1 = torch.allclose(
            dest_cpu, dest_cuda.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False
        )
        self.assertTrue(ret1)

    @staticmethod
    def as_stride2(device):
        inputraw = torch.arange(1, 301).to(device)
        destraw = torch.ones(30, 10).to(device)
        dest1 = destraw.as_strided((30, 5), (5, 1), storage_offset=0)
        input1 = torch.as_strided(
            inputraw, dest1.size(), dest1.stride(), storage_offset=0
        )
        # copy shouldn't change other parts of inputraw
        dest1.copy_(input1)
        return destraw

    def test_as_stride2(self):
        dest_cpu = self.as_stride2("cpu")
        dest_cuda = self.as_stride2("cuda")
        ret1 = torch.allclose(
            dest_cpu, dest_cuda.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False
        )
        self.assertTrue(ret1)


if __name__ == "__main__":
    run_tests()
