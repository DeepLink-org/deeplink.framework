import torch_dipu
import torch
import unittest

from torch_dipu.testing._internal.common_utils import dipu, TestCase, run_tests

class TestSchema(TestCase):

    def test_triu(self):

        xa = torch.randn(3, 3).cuda()
        xb = xa.cpu()

        self.assertTrue(torch.allclose(torch.triu(xa).cpu(),torch.triu(xb)))
        self.assertTrue(torch.allclose(torch.triu(xa, diagonal=1).cpu(),torch.triu(xb, diagonal=1)))
        self.assertTrue(torch.allclose(torch.triu(xa, diagonal=-1).cpu(),torch.triu(xb, diagonal=-1)))

        ya = torch.randn(4, 6).cuda()
        yb = ya.cpu()
        self.assertTrue(torch.allclose(torch.triu(ya, diagonal=1).cpu(),torch.triu(yb, diagonal=1)))
        self.assertTrue(torch.allclose(torch.triu(ya, diagonal=-1).cpu(),torch.triu(yb, diagonal=-1)))

if __name__ == '__main__':
    run_tests()

