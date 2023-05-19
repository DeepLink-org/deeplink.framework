# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

def test_min(tensor):
    r1 = torch.min(tensor.cpu())
    r2 = torch.min(tensor.cuda())
    print(r1, r2)
    assert torch.allclose(r1, r2.cpu())

    r3 = tensor.cpu().min()
    r4 = tensor.cuda().min()
    print(r3, r4)
    assert torch.allclose(r3, r4.cpu())

    print("acc is ok")


def test_partial_min(tensor, dim, keepdim=False, *, out=None):
    r1 = torch.min(input = tensor.cpu(), dim = dim, keepdim=keepdim)
    r2 = torch.min(input = tensor.cuda(), dim = dim, keepdim=keepdim)
    print(r1, r2)
    assert len(r1) == len(r2)
    for i in range(len(r1)):
        assert torch.allclose(r1[i], r2[i].cpu())

    r3 = tensor.cpu().min(dim = dim, keepdim=keepdim)
    r4 = tensor.cuda().min(dim = dim, keepdim=keepdim)
    print(r3, r4)
    for i in range(len(r1)):
        assert torch.allclose(r3[i], r4[i].cpu())

    print("acc is ok")


def test_max(tensor):
    r1 = torch.max(tensor.cpu())
    r2 = torch.max(tensor.cuda())
    print(r1, r2)
    assert torch.allclose(r1, r2.cpu())

    r3 = tensor.cpu().max()
    r4 = tensor.cuda().max()
    print(r3, r4)
    assert torch.allclose(r3, r4.cpu())

    print("acc is ok")


def test_partial_max(tensor, dim, keepdim=False, *, out=None):
    r1 = torch.max(input = tensor.cpu(), dim = dim, keepdim=keepdim)
    r2 = torch.max(input = tensor.cuda(), dim = dim, keepdim=keepdim)
    print(r1, r2)
    assert len(r1) == len(r2)
    for i in range(len(r1)):
        assert torch.allclose(r1[i], r2[i].cpu())

    r3 = tensor.cpu().max(dim = dim, keepdim=keepdim)
    r4 = tensor.cuda().max(dim = dim, keepdim=keepdim)
    print(r3, r4)
    for i in range(len(r1)):
        assert torch.allclose(r3[i], r4[i].cpu())

    print("acc is ok")




test_min(torch.randn(3,4))
test_partial_min(torch.randn(3, 4, 5), dim = 0)
test_partial_min(torch.randn(3, 4, 5), dim = 1)
test_partial_min(torch.randn(3, 4, 5), dim = 1, keepdim=True)

test_max(torch.randn(3,4))
test_partial_max(torch.randn(3, 4, 5), dim = 0)
test_partial_max(torch.randn(3, 4, 5), dim = 1)
test_partial_max(torch.randn(3, 4, 5), dim = 1, keepdim=True)
