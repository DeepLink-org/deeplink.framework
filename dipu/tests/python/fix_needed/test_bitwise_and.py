# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
device = torch.device("dipu")
a = torch.tensor([1, 2, 3]).to(device)
b = torch.tensor([4, 5, 6]).to(device)
c = torch.bitwise_and(a, b)
print(c)
a.bitwise_and_(b)
torch.bitwise_and(a,1)
torch.bitwise_and(a,b,out=c)

torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))