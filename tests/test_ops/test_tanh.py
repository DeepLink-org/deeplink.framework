import torch
import torch_dipu

device = torch.device("dipu")


a = torch.randn(4).to(device)
print(a)
print(torch.tanh(a))
print(torch.Tensor.tanh_(a))
print(a)