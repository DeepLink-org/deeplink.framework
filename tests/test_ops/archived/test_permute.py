import torch
import torch_dipu


device = torch.device("dipu")
x = torch.randn(2, 3, 5).to(device)
print(x)
print(torch.permute(x, (2, 0, 1)))