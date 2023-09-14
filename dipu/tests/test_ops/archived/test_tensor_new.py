# Copyright (c) 2023, DeepLink.

import torch
import torch_dipu

def test_tensor_new(devicestr):
    device = torch.device(devicestr)
    x = torch.randn(10, 10).to(device)
    y = torch.randn(2, 2).to(device)
    z = y.new(x.storage())
    assert torch.allclose(x, z.reshape(10, 10))

test_tensor_new("cpu")
#test_tensor_new("dipu")
