# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")
a = torch.randn(4, 4, 6, 7).to(device)

assert torch.allclose(torch.mean(a, 1, True).cpu(), torch.mean(a.cpu(), 1, True), atol = 1e-3, rtol = 1e-3)
assert torch.allclose(torch.mean(a, (1, 3), True).cpu(), torch.mean(a.cpu(), (1, 3), True), atol = 1e-3, rtol = 1e-3)
assert torch.allclose(torch.mean(a, (1, 3), False).cpu(), torch.mean(a.cpu(), (1, 3), False), atol = 1e-3, rtol = 1e-3)
assert torch.allclose(torch.std(a, 1, True).cpu(), torch.std(a.cpu(), 1, True), atol = 1e-3, rtol = 1e-3)
assert torch.allclose(torch.std(a, (1, 3), True).cpu(), torch.std(a.cpu(), (1, 3), True), atol = 1e-3, rtol = 1e-3)
assert torch.allclose(torch.std(a, (1, 3), False).cpu(), torch.std(a.cpu(), (1, 3), False), atol = 1e-3, rtol = 1e-3)

