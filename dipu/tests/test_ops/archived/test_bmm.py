import unittest
import torch
import torch_dipu

dipu = torch.device('dipu')
cpu = torch.device('cpu')

class TestSchema(unittest.TestCase):
    def test_bmm(self):
        batch1 = torch.randn(3, 3, 2).to(dipu)
        batch2 = torch.randn(3, 2, 4).to(dipu)
        out_dipu = torch.bmm(batch1, batch2)
        out_cpu = torch.bmm(batch1.to(cpu), batch2.to(cpu))
        self.assertTrue(torch.allclose(out_dipu.to(cpu), out_cpu))

if __name__ == '__main__':
    unittest.main()