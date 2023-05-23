import unittest
import torch
import torch_dipu

dipu = torch.device('dipu')
cpu = torch.device('cpu')

class TestNorm(unittest.TestCase):
    
    def test_norm(self):
        x = torch.randn(3, 3, 2).to(dipu)
        y = torch.norm(x)
        z = torch.norm(x.cpu())   
        self.assertTrue(torch.allclose(y.to(cpu), z))
        
        y = torch.norm(x, 2)
        z = torch.norm(x.cpu(), 2)   
        self.assertTrue(torch.allclose(y.to(cpu), z))

if __name__ == '__main__':
    unittest.main()