import torch
import torch_dipu

device = torch.device("dipu")
out_dipu = torch.empty(2, 2).to(device)
print(out_dipu.random_(0, 100))
print(out_dipu.random_())

out_cpu = torch.empty(2, 2)
print(out_cpu.random_(0, 100))
print(out_cpu.random_())