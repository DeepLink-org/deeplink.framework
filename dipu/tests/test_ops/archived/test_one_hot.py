import unittest
import torch
import torch_dipu
dipu = torch.device("dipu")
cpu = torch.device("cpu")

class TestSchema(unittest.TestCase):

    def test_one_hot(self):
        self.assertTrue(torch.allclose(torch.nn.functional.one_hot(torch.arange(0, 5).to(dipu) % 3).to(cpu), torch.nn.functional.one_hot(torch.arange(0, 5).to(cpu) % 3)))
        self.assertTrue(torch.allclose(torch.nn.functional.one_hot(torch.arange(0, 5).to(dipu) % 3, num_classes=5).to(cpu), torch.nn.functional.one_hot(torch.arange(0, 5).to(cpu) % 3, num_classes=5)))
        self.assertTrue(torch.allclose(torch.nn.functional.one_hot(torch.arange(0, 6).to(dipu).view(3,2) % 3).to(cpu), torch.nn.functional.one_hot(torch.arange(0, 6).to(cpu).view(3,2) % 3)))


if __name__ == '__main__':
    unittest.main()