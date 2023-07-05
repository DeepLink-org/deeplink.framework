import torch
import os
os.environ['DIPU_DUMP_OP_ARGS'] = '3'
import torch_dipu
from torch_dipu import dipu


def test_cdist(plist):
    a = torch.randn((5, 3), requires_grad=True)
    b = torch.randn((2, 3, 3), requires_grad=True)

    a_cuda = a.detach().cuda()
    b_cuda = b.detach().cuda()
    a_cuda.requires_grad = True
    b_cuda.requires_grad = True

    for p in plist:
        y = torch.cdist(a, b, p=p)
        y1 = torch.cdist(a_cuda, b_cuda, p=p)
        y.backward(torch.ones_like(y))
        y1.backward(torch.ones_like(y1))
        assert torch.allclose(y, y1.cpu(), atol = 1e-3)
        assert torch.allclose(a.grad, a_cuda.grad.cpu(), atol = 1e-3)
        assert torch.allclose(b.grad, b_cuda.grad.cpu(), atol = 1e-3)

if dipu.vendor_type == "MLU":
    # Currently only 1-norm is supported by camb for the scatter op
    plist = [1]
    test_cdist(plist)
else:
    plist = [1, 2, 0.5, float("inf")]
    test_cdist(plist)
