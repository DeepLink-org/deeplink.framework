# Copyright (c) 2023, DeepLink.
import torch
import numpy as np
import torch_dipu

fn = torch.nn.functional.leaky_relu
fn_ = torch.nn.functional.leaky_relu_

for s in np.arange(0.0, 1.0, 0.05):
    in1 = torch.randn(3, 4, 5, 6)
    in2 = in1.cuda()
    in1.requires_grad = True
    in2.requires_grad = True
    y1 = fn(in1, negative_slope=s)
    y2 = fn(in2, negative_slope=s)
    y1.backward(torch.ones_like(in1))
    y2.backward(torch.ones_like(in2))
    assert torch.allclose(y1, y2.cpu(), atol = 1e-3, rtol=1e-3)
    assert torch.allclose(in1.grad, in2.grad.cpu(), atol = 1e-3, rtol=1e-3)
    assert torch.allclose(fn(in1.clone(), negative_slope=s, inplace = True), fn(in1.cuda(), negative_slope=s, inplace = True).cpu(), atol = 1e-3, rtol=1e-3)
    assert torch.allclose(fn_(in1.clone(), negative_slope=s), fn_(in1.cuda(), negative_slope=s).cpu(), atol = 1e-3, rtol=1e-3)
