import torch_dipu
import torch

def as_stride1():
  device = "cuda"
  # device = "cpu"
  input1 = torch.range(1, 300).to(device)
  dest1 = torch.ones(30, 10).to(device).as_strided((30,5), (5, 1), storage_offset=100)
  ret1 = torch.as_strided(input1, dest1.size(), dest1.stride(), storage_offset=100)
  print(ret1)
  dest1.copy_(ret1)
  print(dest1)

def as_stride2():
  device = "cuda"
  # device = "cpu"
  inputraw = torch.range(1, 300).to(device)
  destraw = torch.ones(30, 10).to(device)
  dest1 = destraw.as_strided((30,5), (5, 1), storage_offset=0)
  input1 = torch.as_strided(inputraw, dest1.size(), dest1.stride(), storage_offset=0)
  print(input1)
  # copy shouldn't change other parts of inputraw 
  dest1.copy_(input1)
  print(dest1)


as_stride1()
as_stride2()