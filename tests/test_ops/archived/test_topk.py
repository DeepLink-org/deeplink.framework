import torch
import torch_dipu

x = torch.arange(1., 6.).double()
print(x, x.shape)
cpu_out = torch.topk(x, 3)
print(cpu_out)
print(cpu_out[0].shape)
value = cpu_out[0].double().cuda().fill_(0)
indices = cpu_out[1].int().cuda().fill_(0)
print(torch.topk(x.cuda(), 3, out = (value, indices)))
print(torch.topk(x.cuda(), 3))