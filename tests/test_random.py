import torch

torch.ops.load_library("../build/libtorch_dipu.so")


out_cuda = torch.empty(2, 2).cuda()
print(out_cuda.random_(0, 100))
print(out_cuda.random_())

out_cpu = torch.empty(2, 2)
print(out_cpu.random_(0, 100))
print(out_cpu.random_())