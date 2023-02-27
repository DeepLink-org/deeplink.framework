import torch

torch.ops.load_library("../build/libtorch_dipu.so")


out_cuda = torch.empty(4).cuda()
print(torch.randperm(4, out = out_cuda, device='cuda'))

out_cpu = torch.empty(4)
print(torch.randperm(4, out = out_cpu))