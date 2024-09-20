import torch
import numpy as np
from torch_dipu.testing._internal.common_utils import TestCase, run_tests

class TestFusedAdamW(TestCase):
    def setUp(self):
        self.weight_shape_list = [(), (16,), (4, 8), (12, 4, 8)]
        self.lr_list = [0.001, 0.01, 0.001, 0.01]
        self.beta1_list = [0.9, 0.9, 0.9, 0.9]
        self.beta2_list = [0.999, 0.999, 0.999, 0.999]
        self.eps_list = [1e-8, 1e-8, 1e-8, 1e-8]
        self.weight_decay_list = [1e-2, 1e-3, 1e-2, 1e-3]
        self.amsgrad_list = [False, False, True, True]
        self.step_list = [2, 3, 4, 5]

    def run_adamw_cpu(self, param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, step, weight_decay, amsgrad):
        torch.optim._functional.adamw(
                [param],
                [param_grad],
                [exp_avg],
                [exp_avg_sq],
                [max_exp_avg_sq],
                [torch.tensor(float(step))],
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                maximize=False,
                )
        return param, exp_avg, exp_avg_sq, max_exp_avg_sq

    def run_adamw_dipu(self, param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, step, weight_decay, amsgrad):
        torch._fused_adamw_(
                [param],
                [param_grad],
                [exp_avg],
                [exp_avg_sq],
                [max_exp_avg_sq],
                [torch.tensor(float(step)).cuda()],
                amsgrad=amsgrad,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                eps=eps,
                maximize=False,
                grad_scale=None,
                found_inf=None,
            )
        return param, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adamw_(self, dtype_):
        for i in range(len(self.weight_shape_list)):
            weight = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            weight_cpu = weight.cpu().to(torch.float32) if dtype_ == torch.float16 else weight.cpu()
            grad = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            grad_cpu = grad.cpu().to(torch.float32) if dtype_ == torch.float16 else grad.cpu()
            m = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            m_cpu = m.cpu().to(torch.float32) if dtype_ == torch.float16 else m.cpu()
            v = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            v_cpu = v.cpu().to(torch.float32) if dtype_ == torch.float16 else v.cpu()
            max_v = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            max_v_cpu = max_v.cpu().to(torch.float32) if dtype_ == torch.float16 else max_v.cpu()
        
            lr = self.lr_list[i]
            beta1 = self.beta1_list[i]
            beta2 = self.beta2_list[i]
            eps = self.eps_list[i]
            weight_decay = self.weight_decay_list[i]
            amsgrad = self.amsgrad_list[i]
            step = self.step_list[i]

            w_new_cpu, m_new_cpu, v_new_cpu, max_v_new_cpu = self.run_adamw_cpu(weight_cpu, grad_cpu, m_cpu, v_cpu, max_v_cpu, lr, beta1, beta2, eps, step, weight_decay, amsgrad)
            w_new, m_new, v_new, max_v_new = self.run_adamw_dipu(weight, grad, m, v, max_v, lr, beta1, beta2, eps, step, weight_decay, amsgrad)
            
            self.assertTrue(
                torch.allclose(w_new.cpu(), w_new_cpu.to(torch.float16) if dtype_ == torch.float16 else w_new_cpu, atol=2e-2 if dtype_ == torch.float16 else 1e-2, rtol=4e-3 if dtype_ == torch.float16 else 2e-3, equal_nan=True),
                torch.allclose(m_new.cpu(), m_new_cpu.to(torch.float16) if dtype_ == torch.float16 else m_new_cpu, atol=2e-2 if dtype_ == torch.float16 else 1e-2, rtol=4e-3 if dtype_ == torch.float16 else 2e-3, equal_nan=True),
                
            )
            self.assertTrue(
                torch.allclose(v_new.cpu(), v_new_cpu.to(torch.float16) if dtype_ == torch.float16 else v_new_cpu, atol=2e-2 if dtype_ == torch.float16 else 1e-2, rtol=4e-3 if dtype_ == torch.float16 else 2e-3, equal_nan=True),
                torch.allclose(max_v_new.cpu(), max_v_new_cpu.to(torch.float16) if dtype_ == torch.float16 else max_v_new_cpu, atol=2e-2 if dtype_ == torch.float16 else 1e-2, rtol=4e-3 if dtype_ == torch.float16 else 2e-3, equal_nan=True)
            )
            
    def test_adamw_fp16_(self):
        self.adamw_(torch.float16)

    def test_adamw_fp32_(self):
        self.adamw_(torch.float32)

if __name__ == "__main__":
    run_tests()

