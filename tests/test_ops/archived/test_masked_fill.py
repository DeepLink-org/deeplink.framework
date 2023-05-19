# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

def test_masked_fill(tensor, mask, value):
    r1 = torch.masked_fill(tensor.cpu().cuda(), mask.cpu(), value.cpu())
    r2 = torch.masked_fill(tensor.cuda(), mask.cuda(), value.cuda())
    print(r1, r2)
    assert torch.allclose(r1, r2.cpu())

def test_masked_fill_scalar(tensor, mask, value):
    r1 = torch.masked_fill(tensor.cpu().cuda(), mask.cpu(), value)
    print(r1)
    r2 = torch.masked_fill(tensor.cuda(), mask.cuda(), value)
    print(r2)
    assert torch.allclose(r1, r2.cpu())

x = torch.randn(3,4)

# camb diopi impl have bug
test_masked_fill_scalar(x, x > 0, 3.14)
#test_masked_fill_scalar(x, x > 0, 3.14)
#test_masked_fill_scalar(x, x < 0, -3.14)
#test_masked_fill(x, x > 0.0, x * 2)