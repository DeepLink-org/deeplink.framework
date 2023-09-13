import torch
import torch_dipu
dipu = torch.device("dipu")
cpu = torch.device("cpu")
input = torch.randn(3, 4)
mask = input.ge(0.5)
cpu = torch.masked_select(input.to(cpu), mask.to(cpu))
dipu = torch.masked_select(input.to(dipu), mask.to(dipu))
assert torch.allclose(cpu, dipu.to(cpu))