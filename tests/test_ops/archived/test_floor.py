import torch
import torch_dipu
import torch.nn as nn

device = torch.device("dipu")
x = torch.randn(4).to(device)
torch.floor(x)
#print(f"y = {torch.floor(x)}")


x = x.cpu()
print(f"y = {torch.floor(x)}")