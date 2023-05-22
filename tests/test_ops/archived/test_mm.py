import torch
import torch_dipu

dipu = torch.device("dipu")
cpu = torch.device("cpu")
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
r1 = torch.mm(mat1.to(dipu), mat2.to(dipu))
r2 = torch.mm(mat1.to(cpu), mat2.to(cpu))
assert torch.allclose(r1.to(cpu), r2)
