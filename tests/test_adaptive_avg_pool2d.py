import torch
import torch.nn.functional as F
import torch_dipu

device = torch.device("dipu")
x = torch.randn(1, 3, 32, 32).to(device)
print(F.adaptive_avg_pool2d(x, (2, 2)))

x = x.cpu()
print(F.adaptive_avg_pool2d(x, (2, 2)))
