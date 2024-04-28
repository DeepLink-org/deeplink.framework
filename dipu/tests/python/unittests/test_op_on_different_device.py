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


if __name__ == "__main__":
    run_tests()
