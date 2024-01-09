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
    
    def test_normal_tensor_tensor2(self):
        dipu = torch.device("dipu")
        mean_val_lst = [-1.2, 3.4, 4, 3, -0.9, -1.6, 3.9, 99, 1000.2, 0.2]
        std_val_lst = [0.1, 3, 5, 8.9, 2.8, 3.2, 4, 1.3, 2.5, 1.1]
        
        mean_tensor = torch.zeros([10, 100000]).to(dipu)
        std_tensor = torch.zeros([10, 100000]).to(dipu)
        
        for i in range(mean_tensor.size()[0]):
            mean_tensor[i].fill_(mean_val_lst[i])
            std_tensor[i].fill_(std_val_lst[i])
        
        out_tensor = torch.normal(mean_tensor, std_tensor)
        
        for i in range(mean_tensor.size()[0]):
            self.assertEqual(out_tensor[i].mean().item(), mean_val_lst[i], prec=0.1)
            self.assertEqual(out_tensor[i].std().item(), std_val_lst[i], prec=0.1)
        
            
    
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
