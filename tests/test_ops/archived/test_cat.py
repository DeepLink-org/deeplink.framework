# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

def test_cat(tensors, dim=0):
    tensors_cpu = []
    tensors_dipu = []
    for tensor in tensors:
        tensors_cpu.append(tensor.cpu())
        tensors_dipu.append(tensor.cuda())

    r1 = torch.cat(tensors_cpu, dim=dim)
    r2 = torch.cat(tensors_dipu, dim=dim).cpu()
    assert torch.allclose(r1, r2)

x = torch.randn(2, 3)
tensors = (x, x, x)

test_cat(tensors, dim = 0)
test_cat(tensors, dim = 1)