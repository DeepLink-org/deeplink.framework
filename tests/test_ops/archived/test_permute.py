import unittest
import torch
import torch_dipu

dipu = torch.device('dipu')
cpu = torch.device('cpu')

class TestSchema(unittest.TestCase):
    def test_permute(self):
        input_dipu = torch.randn(2, 3, 5).to(dipu)
        out_dipu = torch.permute(input_dipu, (2, 0, 1))
        input_cpu = input_dipu.to(cpu)
        out_cpu = torch.permute(input_cpu, (2, 0, 1))
        self.assertTrue(torch.allclose(out_dipu.to(cpu), out_cpu))

if __name__ == '__main__':
    unittest.main()