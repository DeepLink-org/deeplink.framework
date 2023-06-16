import torch
import torch_dipu

src = torch.arange(1, 11).reshape((2, 5))
index = torch.tensor([[0, 1, 2, 0]])
y1 = torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
y2 = torch.zeros(3, 5, dtype=src.dtype).cuda().scatter_(0, index.cuda(), src.cuda())
assert torch.allclose(y1, y2.cpu())

index = torch.tensor([[0, 1, 2], [0, 1, 4]])
torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)

#torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]), 1.23, reduce='multiply') # camb not support
y1 = torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]), 1.23, reduce='add')
y2 = torch.full((2, 4), 2.).cuda().scatter_(1, torch.tensor([[2], [3]]).cuda(), 1.23, reduce='add')
assert torch.allclose(y1, y2.cpu())
