import torch
from torch.nn import functional as F
import torch_dipu
from torch_dipu import diputype

def test_stor1():
  PATH1 = "./test_stor1.pth"

  device = "cuda:0"
  # args is int8,
  args = [[1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0]]
  s1 = torch.UntypedStorage(*args, device=diputype)
  assert s1.device.type == diputype
  #  little endian
  x = torch.arange(1, device = device, dtype=torch.int32)
  x1 = x.new(s1)
  torch.save(x1, PATH1)
  x1 = torch.load(PATH1, map_location="cuda:0")
  print(x1)
  target = torch.tensor([1, 4, 12], device = device, dtype=torch.int32)
  print(target)
  assert(torch.equal(x1, target))

if __name__ == '__main__':
    test_stor1()
