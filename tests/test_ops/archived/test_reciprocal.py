import torch
import torch_dipu
dipu = torch.device("dipu")
cpu = torch.device("cpu")
a = torch.randn(4)
a_dipu = torch.reciprocal(a.to(dipu))
a_cpu = torch.reciprocal(a.to(cpu))
assert torch.allclose(a_cpu, a_dipu.to(cpu))
torch.reciprocal(a.to(dipu),out=a_dipu)
torch.reciprocal(a.to(cpu),out=a_cpu)
assert torch.allclose(a_cpu, a_dipu.to(cpu))