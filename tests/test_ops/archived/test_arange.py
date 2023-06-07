import torch
import torch_dipu
import unittest

from torch_dipu.testing._internal.common_utils import dipu, TestCase, run_tests


class TestSchema(TestCase):

    def test_arange(self):

        #self.assertTrue(torch.allclose(torch.arange(5), torch.arange(5, device='dipu').cpu())) #camb impl has bug
        #self.assertTrue(torch.allclose(torch.arange(5, 20, 1), torch.arange(5, 20, 1, device='dipu').cpu())) #camb impl has bug
        self.assertTrue(torch.allclose(torch.arange(5.), torch.arange(5., device='dipu').cpu()))
        self.assertTrue(torch.allclose(torch.arange(5., 20., 1.0), torch.arange(5., 20.0, 1.0, device='dipu').cpu()))
        self.assertTrue(torch.allclose(torch.arange(5., 20., 0.1), torch.arange(5., 20.0, 0.1, device='dipu').cpu()))


if __name__ == '__main__':
    run_tests()