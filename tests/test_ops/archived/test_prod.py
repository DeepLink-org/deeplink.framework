import torch
import torch_dipu

x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
x_xpu = x.cuda()
print(x_xpu.cpu())

p = torch.prod(x)
p_xpu = torch.prod(x_xpu)
print(p_xpu.cpu())
assert torch.allclose(p_xpu.cpu(), p)

p = torch.prod(x, 0)
p_xpu = torch.prod(x_xpu, 0)
print(p_xpu.cpu())
assert torch.allclose(p_xpu.cpu(), p)

p = torch.prod(x, 1)
p_xpu = torch.prod(x_xpu, 1)
print(p_xpu.cpu())
assert torch.allclose(p_xpu.cpu(), p)

p = torch.prod(x, 0, True)
p_xpu = torch.prod(x_xpu, 0, True)
print(p_xpu.cpu())
assert torch.allclose(p_xpu.cpu(), p)

p = torch.prod(x, 1, True)
p_xpu = torch.prod(x_xpu, 1, True)
print(p_xpu.cpu())
assert torch.allclose(p_xpu.cpu(), p)
