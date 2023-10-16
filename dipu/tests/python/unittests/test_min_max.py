# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMinMax(TestCase):
    def _test_min(self, tensor):
        r1 = torch.min(tensor.cpu())
        r2 = torch.min(tensor.cuda())
        # print(r1, r2)
        assert torch.allclose(r1, r2.cpu())

        r3 = tensor.cpu().min()
        r4 = tensor.cuda().min()
        # print(r3, r4)
        assert torch.allclose(r3, r4.cpu())

        # print("acc is ok")

    def _test_partial_min(self, tensor, dim, keepdim=False, *, out=None):
        r1 = torch.min(input=tensor.cpu(), dim=dim, keepdim=keepdim)
        r2 = torch.min(input=tensor.cuda(), dim=dim, keepdim=keepdim)
        # print(r1, r2)
        assert len(r1) == len(r2)
        for i in range(len(r1)):
            assert torch.allclose(r1[i], r2[i].cpu())

        r3 = tensor.cpu().min(dim=dim, keepdim=keepdim)
        r4 = tensor.cuda().min(dim=dim, keepdim=keepdim)
        # print(r3, r4)
        for i in range(len(r1)):
            assert torch.allclose(r3[i], r4[i].cpu())

        # print("acc is ok")

    def _test_max(self, tensor):
        r1 = torch.max(tensor.cpu())
        r2 = torch.max(tensor.cuda())
        # print(r1, r2)
        assert torch.allclose(r1, r2.cpu())

        r3 = tensor.cpu().max()
        r4 = tensor.cuda().max()
        # print(r3, r4)
        assert torch.allclose(r3, r4.cpu())

        # print("acc is ok")

    def _test_partial_max(self, tensor, dim, keepdim=False, *, out=None):
        r1 = torch.max(input=tensor.cpu(), dim=dim, keepdim=keepdim)
        r2 = torch.max(input=tensor.cuda(), dim=dim, keepdim=keepdim)
        # print(r1, r2)
        assert len(r1) == len(r2)
        for i in range(len(r1)):
            assert torch.allclose(r1[i], r2[i].cpu())

        r3 = tensor.cpu().max(dim=dim, keepdim=keepdim)
        r4 = tensor.cuda().max(dim=dim, keepdim=keepdim)
        # print(r3, r4)
        for i in range(len(r1)):
            assert torch.allclose(r3[i], r4[i].cpu())

        # print("acc is ok")

    def test_min(self):
        self._test_min(torch.randn(3, 4))

    def test_partial_min(self):
        self._test_partial_min(torch.randn(3, 4, 5), dim=0)
        self._test_partial_min(torch.randn(3, 4, 5), dim=1)
        self._test_partial_min(torch.randn(3, 4, 5), dim=1, keepdim=True)

    def test_max(self):
        self._test_max(torch.randn(3, 4))

    def test_partial_max(self):
        self._test_partial_max(torch.randn(3, 4, 5), dim=0)
        self._test_partial_max(torch.randn(3, 4, 5), dim=1)
        self._test_partial_max(torch.randn(3, 4, 5), dim=1, keepdim=True)


if __name__ == "__main__":
    run_tests()
