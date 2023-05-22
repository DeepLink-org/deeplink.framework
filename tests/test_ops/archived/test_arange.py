import unittest
import torch
import torch_dipu

dipu = torch.device('dipu')
cpu = torch.device('cpu')

class TestSchema(unittest.TestCase):
    def test_arange(self):
        # 无选择的版本
        dipu_out = torch.empty(5, device=dipu).to(dipu)
        cpu_out = torch.empty(5, device=cpu).to(cpu)
        dipu_a = torch.arange(5, device=dipu).to(dipu)
        cput_a = torch.arange(5, device=cpu).to(cpu)
        a_dipu_out = torch.arange(5, out=dipu_out, device=dipu).to(dipu)
        a_cpu_out = torch.arange(5, out=cpu_out, device=cpu).to(cpu)
        print(dipu_a, cput_a)
        self.assertTrue(torch.allclose(dipu_a.to(cpu), cput_a))
        self.assertTrue(torch.allclose(a_dipu_out.to(cpu), a_cpu_out))

        dipu_b = torch.arange(1, 6, device=dipu).to(dipu)
        cpu_b = torch.arange(1, 6, device=cpu).to(cpu)
        b_dipu_out = torch.arange(1, 6, out=dipu_out, device=dipu).to(dipu)
        b_cpu_out = torch.arange(1, 6, out=cpu_out, device=cpu).to(cpu)
        self.assertTrue(torch.allclose(dipu_b.to(cpu), cpu_b))
        self.assertTrue(torch.allclose(b_dipu_out.to(cpu), b_cpu_out))
        
        dipu_out = torch.empty(3, device=dipu).to(dipu)
        cpu_out = torch.empty(3, device=cpu).to(cpu)
        dipu_c = torch.arange(start = 1, end = 6, step = 2, device = dipu).to(dipu)
        cpu_c = torch.arange(start = 1, end = 6, step = 2, device = cpu).to(cpu)
        c_dipu_out = torch.arange(start = 1, end = 6, step = 2, out = dipu_out, device = dipu).to(dipu)
        c_cpu_out = torch.arange(start = 1, end = 6, step = 2, out = cpu_out, device = cpu).to(cpu)
        self.assertTrue(torch.allclose(dipu_c.to(cpu), cpu_c))
        self.assertTrue(torch.allclose(c_dipu_out.to(cpu), c_cpu_out))
    
        
if __name__ == '__main__':
    unittest.main()

