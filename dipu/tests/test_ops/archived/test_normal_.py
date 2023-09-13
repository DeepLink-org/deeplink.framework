import torch
import torch_dipu

dipu = torch.device("dipu")
cpu = torch.device("cpu")
x = torch.Tensor([[1,2,3,4.9]]).to(dipu)
x.normal_(mean=0.0, std=1.0)


print(x)