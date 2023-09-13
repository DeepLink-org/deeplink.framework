# Copyright (c) 2023, DeepLink.
import unittest
import torch
from torch.nn import functional as F
import torch_dipu


dipu = torch.device("dipu")
cpu = torch.device("cpu")


class TestSchema(unittest.TestCase):

    def test_where_1(self):
        x = torch.randn(3, 2)
        z1 = torch.where(x.to(dipu) > 0, 2.0, 0.0)
        z2 = torch.where(x > 0, 2.0, 0.0)
        self.assertTrue(torch.allclose(z1.to(cpu),z2))
    
    def test_where_2(self):
        x = torch.randn(3, 2)
        y = torch.ones(3, 1)
        z1 = torch.where(x.to(dipu) > 0, x.to(dipu), y.to(dipu))
        z2 = torch.where(x > 0, x, y)
        self.assertTrue(torch.allclose(z1.to(cpu),z2))

    def test_where_3(self):
        x = torch.randn(3, 1,1)
        y = torch.ones(3, 2,1)
        z = torch.zeros(3, 1,2)

        z1 = torch.where(x.to(dipu) > 0, y.to(dipu), z.to(dipu))
        z2 = torch.where(x > 0, y, z)
        self.assertTrue(torch.allclose(z1.to(cpu),z2))

if __name__ == '__main__':
    unittest.main()


