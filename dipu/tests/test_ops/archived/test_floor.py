import unittest
import torch
import torch_dipu

dipu = torch.device("dipu")
cpu = torch.device("cpu")



class TestSchema(unittest.TestCase):

    def test_floor(self):
        x = torch.randn(4)
        z1 = torch.floor(x)
        z2 = torch.floor(x.to(dipu))
        self.assertTrue(torch.allclose(z1,z2.to(cpu)))


if __name__ == '__main__':
    unittest.main()
