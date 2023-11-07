# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestCast(TestCase):
    def _test_cat(self, tensors, dim=0):
        tensors_cpu = []
        tensors_dipu = []
        for tensor in tensors:
            tensors_cpu.append(tensor.cpu())
            tensors_dipu.append(tensor.cuda())

        r1 = torch.cat(tensors_cpu, dim=dim)
        r2 = torch.cat(tensors_dipu, dim=dim).cpu()
        self.assertEqual(r1, r2)

    def test_cast(self):
        x = torch.randn(2, 3)
        tensors = (x, x, x)

        self._test_cat(tensors, dim=0)
        self._test_cat(tensors, dim=1)

    def test_cat2(self):
        device = torch.device("dipu")
        data = torch.randn(8, 8732, dtype=torch.float64).to(device)
        data1 = data[:, :5776]
        data2 = data[:, 5776:]
        res = torch.cat([data1, data2], -1)
        self.assertEqual(res.cpu(), data.cpu())


if __name__ == "__main__":
    run_tests()
