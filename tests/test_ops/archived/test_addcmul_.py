import torch
import torch_dipu

device = torch.device("dipu")



t = torch.tensor([0.,0.,0.]).to(device)
t1 = torch.tensor([[1.],[2.] ,[3.]]).to(device)
t2 = torch.tensor([1.,2.,3.]).to(device)

print(t)
print(t1)
print(t2)

print(torch.addcmul(t, t1, t2, value=0.1))
print(torch.Tensor.addcmul_(t, t2, t2, value=0.1))
print(t)