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

def test_cat2():
    device = torch.device("dipu")
    data = torch.randn(8, 8732, dtype=torch.float64).to(device)
    data1 = data[:, :5776]
    data2 = data[:, 5776:]
    res = torch.cat([data1, data2], -1)
    assert torch.allclose(res.cpu(), data.cpu())

x = torch.randn(2, 3)
tensors = (x, x, x)

test_cat(tensors, dim = 0)
test_cat(tensors, dim = 1)

test_cat2()