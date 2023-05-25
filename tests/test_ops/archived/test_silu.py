import torch
import torch_dipu

dipu = torch.device("dipu")
cpu = torch.device("cpu")
a = torch.randn(2)


a_cpu = a.to(cpu)
a_dipu = a.to(dipu)
silu_cpu = torch.nn.SiLU()(a_cpu)
silu_dipu = torch.nn.SiLU()(a_dipu)
assert torch.allclose(silu_cpu, silu_dipu.to(cpu))
torch.nn.SiLU(inplace = True)(a_cpu)
torch.nn.SiLU(inplace = True)(a_dipu)
assert torch.allclose(a_cpu, a_dipu.to(cpu))
print(a_cpu,a_dipu.to(cpu))