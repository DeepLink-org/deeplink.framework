# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu



def _test_baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    cpu_result = torch.baddbmm(input.cpu(), batch1.cpu(), batch2.cpu(), beta = beta, alpha = alpha)
    device_result = torch.baddbmm(input.cuda(), batch1.cuda(), batch2.cuda(), beta = beta, alpha = alpha)
    assert torch.allclose(cpu_result, device_result.cpu())

    cpu_result = input.cpu().baddbmm(batch1.cpu(), batch2.cpu(), beta = beta, alpha = alpha)
    device_result = input.cuda().baddbmm(batch1.cuda(), batch2.cuda(), beta = beta, alpha = alpha)
    assert torch.allclose(cpu_result, device_result.cpu())

    cpu_result = input.cpu().baddbmm_(batch1.cpu(), batch2.cpu(), beta = beta, alpha = alpha)
    device_result = input.cuda().baddbmm_(batch1.cuda(), batch2.cuda(), beta = beta, alpha = alpha)
    assert torch.allclose(cpu_result, device_result.cpu())


def test_baddbmm1():
    M = torch.randn(10, 3, 5).cuda()
    batch1 = torch.randn(10, 3, 4).cuda()
    batch2 = torch.randn(10, 4, 5).cuda()
    _test_baddbmm(M, batch1, batch2)

def test_baddbmm2():
    M = torch.randn(10, 3, 5).cuda()
    batch1 = torch.randn(10, 3, 4).cuda()
    batch2 = torch.randn(10, 4, 5).cuda()
    _test_baddbmm(M, batch1, batch2, alpha=2, beta=3)