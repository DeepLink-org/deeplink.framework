import torch
import torch_dipu
dipu = torch.device("dipu")
cpu = torch.device("cpu")
input = torch.randn(4, 4)
dipu = torch.argmax(input.to(dipu))
cpu = torch.argmax(input.to(cpu))
assert torch.allclose(cpu, dipu.to(cpu))