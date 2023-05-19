# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

device = torch.device("dipu")

# t = torch.randn(1, 3).to(device)
# t1 = torch.randn(3, 1).to(device)
# t2 = torch.randn(1, 3).to(device)

t = torch.tensor([0.,0.,0.]).to(device)
t1 = torch.tensor([[1.],[2.] ,[3.]]).to(device)
t2 = torch.tensor([1.,2.,3.]).to(device)

print(t)
print(t1)
print(t2)

print(torch.addcdiv(t, t1, t2, value=0.1))
print(torch.Tensor.addcdiv_(t, t2, t2, value=0.1))
print(t)