# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, skipOn


class TestScatter(TestCase):
    def setUp(self):
        self.src = torch.arange(1, 11).reshape((2, 5))
        self.index = torch.tensor([[0, 1, 2, 0]])

    def test_scatter__tensor_src(self):
        y1 = torch.zeros(3, 5, dtype=self.src.dtype).scatter_(0, self.index, self.src)
        y2 = (
            torch.zeros(3, 5, dtype=self.src.dtype)
            .cuda()
            .scatter_(0, self.index.cuda(), self.src.cuda())
        )
        self.assertEqual(y1, y2.cpu())

    def test_scatter__tensor_src_add(self):
        # The reduction operation of multiply is not supported by camb
        y1 = torch.zeros(3, 5, dtype=self.src.dtype).scatter_(
            0, self.index, self.src, reduce="add"
        )
        y2 = (
            torch.zeros(3, 5, dtype=self.src.dtype)
            .cuda()
            .scatter_(0, self.index.cuda(), self.src.cuda(), reduce="add")
        )
        self.assertEqual(y1, y2.cpu())

    @skipOn("MLU", "The reduction operation of multiply is not supported by camb")
    def test_scatter__tensor_src_multiply(self):
        y1 = torch.zeros(3, 5, dtype=self.src.dtype).scatter_(
            0, self.index, self.src, reduce="multiply"
        )
        y2 = (
            torch.zeros(3, 5, dtype=self.src.dtype)
            .cuda()
            .scatter_(0, self.index.cuda(), self.src.cuda(), reduce="multiply")
        )
        self.assertEqual(y1, y2.cpu())

    def test_scatter__scalar_src(self):
        y1 = torch.full((2, 4), 2.0).scatter_(1, torch.tensor([[2], [3]]), 1.23)
        y2 = (
            torch.full((2, 4), 2.0)
            .cuda()
            .scatter_(1, torch.tensor([[2], [3]]).cuda(), 1.23)
        )
        self.assertEqual(y1, y2.cpu())

    def test_scatter__scalar_src_add(self):
        y1 = torch.full((2, 4), 2.0).scatter_(
            1, torch.tensor([[2], [3]]), 1.23, reduce="add"
        )
        y2 = (
            torch.full((2, 4), 2.0)
            .cuda()
            .scatter_(1, torch.tensor([[2], [3]]).cuda(), 1.23, reduce="add")
        )
        self.assertEqual(y1, y2.cpu())

    @skipOn("MLU", "The reduction operation of multiply is not supported by camb")
    def test_scatter__scalar_src_multiply(self):
        y1 = torch.full((2, 4), 2.0).scatter_(
            1, torch.tensor([[2], [3]]), 1.23, reduce="multiply"
        )
        y2 = (
            torch.full((2, 4), 2.0)
            .cuda()
            .scatter_(1, torch.tensor([[2], [3]]).cuda(), 1.23, reduce="multiply")
        )
        self.assertEqual(y1, y2.cpu())

    def test_scatter_add_out(self):
        y1 = torch.full((3, 5), 2).scatter_add(0, self.index, self.src)
        y2 = (
            torch.full((3, 5), 2)
            .cuda()
            .scatter_add(0, self.index.cuda(), self.src.cuda())
        )
        self.assertEqual(y1, y2.cpu())


if __name__ == "__main__":
    run_tests()
