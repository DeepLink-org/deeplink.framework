# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

x = torch.arange(200).reshape(4,50).cuda()
y = torch.arange(200).reshape(4,50)

assert torch.allclose(x.cpu(), y)
assert torch.allclose(x.double().cpu(), y.double())
assert torch.allclose(x.int().cpu(), y.int())
assert torch.allclose(x.long().cpu(), y.long())