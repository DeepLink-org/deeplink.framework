# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestOpOnDifferentDevice(TestCase):
    def _test_add_on_device(self, device_index):
        torch.cuda.set_device(device_index)
        x = torch.randn(3, 4).cuda()
        y = x.cpu()

        self.assertEqual((x + x).cpu(), y + y)
        self.assertEqual((x + x).cpu(), y + y)

        x.add_(3)
        y.add_(3)
        self.assertEqual(x.cpu(), y)

        x.add_(3)
        y.add_(3)
        self.assertEqual(x.cpu(), y)

        x.add_(torch.ones_like(x))
        y.add_(torch.ones_like(y))
        self.assertEqual(x.cpu(), y)

    def test_op_on_different_device(self):
        for i in range(torch.cuda.device_count()):
            self._test_add_on_device(i)

    def test_allocate_tensor_on_different_device(self):
        device_count = torch.cuda.device_count()
        for index in range(device_count):
            shape = [2, 4, 5, 6]
            dtype = torch.float
            a = torch.empty(shape, dtype=dtype, device="cuda:" + str(index))
            assert a.device.index == index

            b = torch.ones(shape, dtype=dtype, device="cuda:" + str(index))
            assert b.device.index == index

            c = torch.zeros(shape, dtype=dtype, device="cuda:" + str(index))
            assert c.device.index == index

            d = torch.full(shape, index, dtype=dtype, device="cuda:" + str(index))
            assert d.device.index == index


if __name__ == "__main__":
    run_tests()
