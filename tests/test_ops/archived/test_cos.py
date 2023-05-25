import unittest
import torch
import torch_dipu

dipu = torch.device('dipu')
cpu = torch.device('cpu')

class TestSchema(unittest.TestCase):
    def test_cos(self):
        input_dipu = torch.Tensor([[1, 2, 4, 5,0,-1]]).to(dipu)
        input_cpu = torch.Tensor([[1, 2, 4, 5,0,-1]]).to(cpu)
        out_dipu = torch.cos(input_dipu)
        out_cpu = torch.cos(input_cpu)
        self.assertTrue(torch.allclose(out_dipu.to(cpu), out_cpu))

        out_dipu = torch.empty_like(input_dipu).to(dipu)
        out_cpu = torch.empty_like(input_cpu).to(cpu)
        torch.cos(input_cpu, out = out_cpu)
        torch.cos(input_dipu, out = out_dipu)
        self.assertTrue(torch.allclose(out_dipu.to(cpu), out_cpu))
        
        input_dipu.cos_()
        input_cpu.cos_()
        self.assertTrue(torch.allclose(input_dipu.to(cpu), input_cpu))



if __name__ == '__main__':
    unittest.main()