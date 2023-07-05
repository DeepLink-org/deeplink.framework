import torch
import torch_dipu
from torch_dipu import dipu


if dipu.vendor_type == "MLU":
    # Currently only 1-norm is supported by camb for the scatter op
    a = torch.randn((5,3),requires_grad = True)
    b = torch.randn((2,3,3),requires_grad = True)
    
    a_cuda=a.detach().cuda()
    b_cuda=b.detach().cuda()
    a_cuda.requires_grad = True
    b_cuda.requires_grad = True
    
    y = torch.cdist(a, b, p=1)
    y1 = torch.cdist(a_cuda, b_cuda, p=1)
    y.backward(torch.ones_like(y))
    y1.backward(torch.ones_like(y1))
    assert torch.allclose(y, y1.cpu())
    assert torch.allclose(a.grad, a_cuda.grad.cpu())
    assert torch.allclose(b.grad, b_cuda.grad.cpu())
else:
    a = torch.randn((5,3),requires_grad = True)
    b = torch.randn((2,3,3),requires_grad = True)
    
    a_cuda=a.detach().cuda()
    b_cuda=b.detach().cuda()
    a_cuda.requires_grad = True
    b_cuda.requires_grad = True
    
    y = torch.cdist(a, b, p=2)
    y1 = torch.cdist(a_cuda, b_cuda, p=2)
    y.backward(torch.ones_like(y))
    y1.backward(torch.ones_like(y1))
    assert torch.allclose(y, y1.cpu())
    assert torch.allclose(a.grad, a_cuda.grad.cpu())
    assert torch.allclose(b.grad, b_cuda.grad.cpu())