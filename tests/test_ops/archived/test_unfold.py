import torch_dipu
import torch

def unfoldtest1():
  device = "cuda"
  x0 = torch.arange(0, 12).reshape((3, 4))
  x = x0.to(dtype=torch.float32, device = device)
  x.requires_grad = True
  res = x.unfold(0, 2, 1)
  print(res.size(), res.stride())
  grad_raw = torch.ones_like(res)
  res.backward(grad_raw)
  print(x.grad)


unfoldtest1()