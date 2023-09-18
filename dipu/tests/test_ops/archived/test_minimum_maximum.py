import unittest
import torch
import torch_dipu
dipu = torch.device("dipu")
cpu = torch.device("cpu")

class TestSchema(unittest.TestCase):

    def test_minimum(self):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4))
        r_dipu = torch.minimum(a.to(dipu), b.to(dipu))
        r_cpu = torch.minimum(a.to(cpu), b.to(cpu))
        self.assertTrue(torch.allclose(r_dipu.to(cpu), r_cpu))

    def test_maximum(self):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4))
        r_dipu = torch.maximum(a.to(dipu), b.to(dipu))
        r_cpu = torch.maximum(a.to(cpu), b.to(cpu))
        self.assertTrue(torch.allclose(r_dipu.to(cpu), r_cpu))

if __name__ == '__main__':
    unittest.main()