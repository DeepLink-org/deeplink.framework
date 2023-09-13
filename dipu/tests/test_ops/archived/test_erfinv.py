import torch
import torch_dipu
for i in range(100):
    x = torch.randn((i + 1, i + 2))
    torch.allclose(torch.special.erfinv(x), torch.special.erfinv(x.cuda()).cpu(), atol = 1e-3, rtol = 1e-3, equal_nan = True)