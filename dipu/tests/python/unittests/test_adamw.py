import torch
import numpy as np
from torch_dipu.testing._internal.common_utils import (
    TestCase,
    run_tests,
    onlyOn,
    skipOn,
)
#`fused=True` requires all the params to be floating point Tensors of supported devices: ['cuda', 'xpu', 'privateuseone'].
#So we use fused=False and cuda results to compare with fused torch_dipu results.

class TestFusedAdamW(TestCase):
    def setUp(self):
        self.weight_shape_list = [(), (16,), (4, 8), (12, 4, 8)]
        self.lr_list = [0.001, 0.01, 0.001, 0.01]
        self.beta1_list = [0.9, 0.9, 0.9, 0.9]
        self.beta2_list = [0.999, 0.999, 0.999, 0.999]
        self.eps_list = [1e-8, 1e-8, 1e-8, 1e-8]
        self.weight_decay_list = [1e-2, 1e-3, 1e-2, 1e-3]
        self.amsgrad_list = [False, False, True, True]

    def run_adamw_cpu(
        self,
        param,
        param_grad,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        amsgrad,
    ):
        param.grad = param_grad
        optimizer = torch.optim.AdamW(params = [param],
                                      lr = lr,
                                      betas = (beta1,beta2),
                                      eps=eps,
                                      weight_decay=weight_decay,
                                      amsgrad = amsgrad,
                                      fused = False)
        optimizer.step()
        state_index = 0
        exp_avg = optimizer.state_dict()["state"][state_index]["exp_avg"]
        exp_avg_sq = optimizer.state_dict()["state"][state_index]["exp_avg_sq"]
        return param, exp_avg, exp_avg_sq
    
    def run_adamw_dipu(
        self,
        param,
        param_grad,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        amsgrad,
    ):
        param.grad = param_grad
        optimizer = torch.optim.AdamW(params = [param],
                                      lr = lr,
                                      betas = (beta1,beta2),
                                      eps=eps,
                                      weight_decay=weight_decay,
                                      amsgrad = amsgrad,
                                      fused = True)
        optimizer.step()
        state_index = 0
        exp_avg = optimizer.state_dict()["state"][state_index]["exp_avg"]
        exp_avg_sq = optimizer.state_dict()["state"][state_index]["exp_avg_sq"]
        return param, exp_avg, exp_avg_sq

    def adamw_(self, dtype_):
        for i in range(len(self.weight_shape_list)):
            weight = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            weight_cpu = (
                weight.cpu().to(torch.float32)
                if dtype_ == torch.float16
                else weight.cpu()
            )
            weight_fused_cpu = weight_cpu.clone().detach()
            grad = torch.randn(self.weight_shape_list[i], dtype=dtype_).cuda()
            grad_cpu = (
                grad.cpu().to(torch.float32) if dtype_ == torch.float16 else grad.cpu()
            )
            grad_fused_cpu = grad_cpu.clone().detach() 

            lr = self.lr_list[i]
            beta1 = self.beta1_list[i]
            beta2 = self.beta2_list[i]
            eps = self.eps_list[i]
            weight_decay = self.weight_decay_list[i]
            amsgrad = self.amsgrad_list[i]

            w_new_cpu, m_new_cpu, v_new_cpu = self.run_adamw_cpu(
                weight_cpu,
                grad_cpu,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
            )
            w_new, m_new, v_new= self.run_adamw_dipu(
                weight,
                grad,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
            )

            self.assertTrue(
                torch.allclose(
                    w_new.cpu(),
                    (
                        w_new_cpu.to(torch.float16)
                        if dtype_ == torch.float16
                        else w_new_cpu
                    ),
                    atol=2e-2 if dtype_ == torch.float16 else 1e-2,
                    rtol=4e-3 if dtype_ == torch.float16 else 2e-3,
                    equal_nan = False,
                ),
            )
             
            self.assertTrue(
                torch.allclose(
                    m_new.cpu(),
                    (
                        m_new_cpu.to(torch.float16)
                        if dtype_ == torch.float16
                        else m_new_cpu
                    ),
                    atol=2e-2 if dtype_ == torch.float16 else 1e-2,
                    rtol=4e-3 if dtype_ == torch.float16 else 2e-3,
                    equal_nan = False,
                ),
            )
            self.assertTrue(
                torch.allclose(
                    v_new.cpu(),
                    (
                        v_new_cpu.to(torch.float16)
                        if dtype_ == torch.float16
                        else v_new_cpu
                    ),
                    atol=2e-2 if dtype_ == torch.float16 else 1e-2,
                    rtol=4e-3 if dtype_ == torch.float16 else 2e-3,
                    equal_nan = False,
                ),
            )

    @skipOn(
        ["MLU", "NPU", "MUXI", "GCU", "DROPLET", "SUPA", "KLX"],
        "The adamw fusion operator has not yet been connected to the dipu of these chips, and the chip name can be removed from the above list after being added later",
    )
    def test_adamw_fp16_(self):
        self.adamw_(torch.float16)

    @skipOn(
        ["MLU", "NPU", "MUXI", "GCU", "DROPLET", "SUPA", "KLX"],
        "The adamw fusion operator has not yet been connected to the dipu of these chips, and the chip name can be removed from the above list after being added later",
    )
    def test_adamw_fp32_(self):
        self.adamw_(torch.float32)


if __name__ == "__main__":
    run_tests()
