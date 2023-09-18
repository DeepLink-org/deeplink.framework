# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

def test_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    r1 = torch.unique(input.cpu(), sorted, return_inverse, return_counts, dim)
    r2 = torch.unique(input.cuda(), sorted, return_inverse, return_counts, dim)
    assert len(r1) == len(r2)
    for i in range(len(r1)):
        assert torch.allclose(r1[i], r2[i].cpu())

    print("acc is ok")

x = torch.randn(3,4,2,3)
test_unique(x, return_inverse = False)
test_unique(x, return_inverse = True)
test_unique(x, return_inverse = False, return_counts=True)
test_unique(x, return_inverse = True, return_counts=True)
test_unique(x, return_inverse = False, dim=0)
test_unique(x, return_inverse = True, dim=0)
test_unique(x, return_inverse = False, return_counts=True, dim=0)
test_unique(x, return_inverse = True, return_counts=True, dim=0)
test_unique(x, return_inverse = True, return_counts=True, dim=2)
