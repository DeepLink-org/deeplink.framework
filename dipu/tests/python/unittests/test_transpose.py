# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestTranspose(TestCase):
    def check_tensor(self, x, gold, shape, stride):
        # print(x, x.shape, x.stride())
        self.assertEqual(x, gold)
        self.assertEqual(x.shape, shape)
        self.assertEqual(x.stride(), stride)

    def test_transpose(self):
        x = torch.randn(2, 3)
        gold = x.T.clone()
        self.check_tensor(x, x, torch.Size([2, 3]), (3, 1))

        y = torch.transpose(x, 0, 1)
        self.check_tensor(y, gold, torch.Size([3, 2]), (1, 3))

        y = torch.transpose(x.cuda(), 0, 1)
        self.check_tensor(y, gold, torch.Size([3, 2]), (1, 3))

        y = x.clone()
        y.transpose_(0, 1)
        self.check_tensor(y, gold, torch.Size([3, 2]), (1, 3))

        # have problem on camb?
        y = x.clone().cuda()
        y.transpose_(0, 1)
        self.check_tensor(y, gold, torch.Size([3, 2]), (1, 3))

        temp = torch.randn([16, 836, 32])
        a = temp.to("dipu").transpose(0, 1).contiguous()
        b = temp.cpu().transpose(0, 1).contiguous()
        ret1 = torch.allclose(b, a.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False)
        self.assertTrue(ret1)


if __name__ == "__main__":
    run_tests()
