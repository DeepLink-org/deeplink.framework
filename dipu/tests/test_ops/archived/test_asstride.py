import torch_dipu
import torch

def as_stride1(device):
  input1 = torch.range(1, 300).to(device)
  dest1 = torch.ones(30, 10).to(device).as_strided((30,5), (5, 1), storage_offset=100)
  ret1 = torch.as_strided(input1, dest1.size(), dest1.stride(), storage_offset=100)
  dest1.copy_(ret1)
  return dest1

def test_as_stride1():
  dest_cpu = as_stride1("cpu")
  dest_cuda = as_stride1("cuda")
  ret1 = torch.allclose(dest_cpu, dest_cuda.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False)
  assert(ret1)


def as_stride2(device):
  inputraw = torch.range(1, 300).to(device)
  destraw = torch.ones(30, 10).to(device)
  dest1 = destraw.as_strided((30,5), (5, 1), storage_offset=0)
  input1 = torch.as_strided(inputraw, dest1.size(), dest1.stride(), storage_offset=0)
  # copy shouldn't change other parts of inputraw 
  dest1.copy_(input1)
  print(destraw)
  return destraw


def test_as_stride2():
  dest_cpu = as_stride2("cpu")
  dest_cuda = as_stride2("cuda")
  ret1 = torch.allclose(dest_cpu, dest_cuda.cpu(), rtol=1e-05, atol=1e-08, equal_nan=False)
  assert(ret1)

test_as_stride1()
test_as_stride2()