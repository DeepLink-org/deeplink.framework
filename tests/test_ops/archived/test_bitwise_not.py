import torch
import torch_dipu

device = torch.device("dipu")
a = torch.tensor([1, 2, 3]).to(device)

a.bitwise_not_()
a.bitwise_not()
b = torch.bitwise_not(a)