# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestNormal(TestCase):
    def test_normal_(self):
        dipu = torch.device("dipu")
        x = torch.zeros(1000000).to(dipu)
        x.normal_(mean=0.0, std=1.0)
        self.assertEqual(x.mean().item(), 0.0, prec=0.01)
        self.assertEqual(x.std().item(), 1.0, prec=0.01)
    
    def test_normal_scalar_scalar(self):
        dipu = torch.device("dipu")
        mean_val = 0
        std_val = 1
        out_size = [1000000]
        out_tensor = torch.normal(mean=mean_val, std=std_val, size=out_size, device=dipu)
        self.assertEqual(out_tensor.mean().item(), mean_val, prec=0.01)
        self.assertEqual(out_tensor.std().item(), std_val, prec=0.01)
        
    def test_normal_tensor_tensor(self):
        dipu = torch.device("dipu")
        mean_val = 0
        std_val = 1
        mean_tensor = torch.zeros([1000, 1000]).fill_(mean_val).to(dipu)
        std_tensor = torch.zeros([1000, 1000]).fill_(std_val).to(dipu)
        out_tensor = torch.normal(mean_tensor, std_tensor)
        self.assertEqual(out_tensor.mean().item(), mean_val, prec=0.01)
        self.assertEqual(out_tensor.std().item(), std_val, prec=0.01)
    
    def test_normal_tensor_scalar(self):
        dipu = torch.device("dipu")
        mean_val = 0
        std_val = 1
        mean_tensor = torch.zeros([1000, 1000]).fill_(mean_val).to(dipu)
        out_tensor = torch.normal(mean_tensor, std_val)
        self.assertEqual(out_tensor.mean().item(), mean_val, prec=0.01)
        self.assertEqual(out_tensor.std().item(), std_val, prec=0.01)
    
    def test_normal_scalar_tensor(self):
        dipu = torch.device("dipu")
        mean_val = 0
        std_val = 1
        std_tensor = torch.zeros([1000, 1000]).fill_(std_val).to(dipu)
        out_tensor = torch.normal(mean_val, std_tensor)
        self.assertEqual(out_tensor.mean().item(), mean_val, prec=0.01)
        self.assertEqual(out_tensor.std().item(), std_val, prec=0.01)


if __name__ == "__main__":
    run_tests()
