# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMockCudaTensor(TestCase):
    def test_mock_cudatensor(self):
        tensor_types = {
            torch.float: torch.cuda.FloatTensor,
            torch.double: torch.cuda.DoubleTensor,
            torch.half: torch.cuda.HalfTensor,
            torch.long: torch.cuda.LongTensor,
            torch.int: torch.cuda.IntTensor,
            torch.short: torch.cuda.ShortTensor,
            torch.uint8: torch.cuda.ByteTensor,
            torch.int8: torch.cuda.CharTensor,
            torch.bool: torch.cuda.BoolTensor,
            }
        for dtype, tensor_type in tensor_types.items():
            tensor = tensor_type([1.])
            self.assertEqual(tensor.dtype, dtype)
            self.assertTrue(tensor.is_cuda)
            self.assertTrue(isinstance(tensor, tensor_type))


if __name__ == "__main__":
    run_tests()
