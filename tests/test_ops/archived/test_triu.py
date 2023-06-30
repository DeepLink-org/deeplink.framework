import torch_dipu
import torch

xa = torch.randn(3, 3).cuda()
xb = xa.cpu()

torch.allclose(torch.triu(xa).cpu(),torch.triu(xb))
torch.allclose(torch.triu(xa, diagonal=1).cpu(),torch.triu(xb, diagonal=1))
torch.allclose(torch.triu(xa, diagonal=-1).cpu(),torch.triu(xb, diagonal=-1))

ya = torch.randn(4, 6).cuda()
yb = ya.cpu()
torch.allclose(torch.triu(ya, diagonal=1).cpu(),torch.triu(yb, diagonal=1))
torch.allclose(torch.triu(ya, diagonal=-1).cpu(),torch.triu(yb, diagonal=-1))

