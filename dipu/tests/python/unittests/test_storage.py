# Copyright (c) 2023, DeepLink.
import tempfile
import os
import torch
import torch_dipu
from torch_dipu import diputype
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestStorage(TestCase):
    def test_stor1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "test_stor1.pth")
            # stor_shared1 = torch.UntypedStorage._new_shared(3, device="cpu")
            # print(stor_shared1)
            device = "cuda:0"
            # args is int8,
            args = [[1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0]]
            s1 = torch.UntypedStorage(*args, device=diputype)
            self.assertEqual(s1.device.type, diputype)
            #  little endian
            x = torch.arange(1, device=device, dtype=torch.int32)
            x1 = x.new(s1)
            torch.save(x1, path1)
            x1 = torch.load(path1, map_location="cuda:0")
            # print(x1)
            target = torch.tensor([1, 4, 12], device=device, dtype=torch.int32)
            # print(target)
            self.assertEqual(x1, target, prec=0)

            snew = s1.resize_(0)
            self.assertEqual(snew.size(), 0)
            self.assertTrue(x1.is_dipu)
            self.assertTrue(x1.is_cuda)


if __name__ == "__main__":
    run_tests()
