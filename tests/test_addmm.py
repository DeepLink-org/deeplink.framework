import torch
import torch_dipu

device = torch.device("dipu")
M = torch.randn(2, 3).to(device)
mat1 = torch.randn(2, 3).to(device)
mat2 = torch.randn(3, 3).to(device)
print(torch.addmm(M, mat1, mat2))

M = M.cpu()
mat1 = mat1.cpu()
mat2 = mat2.cpu()
print(torch.addmm(M, mat1, mat2))