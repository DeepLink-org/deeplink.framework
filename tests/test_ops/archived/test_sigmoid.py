import torch
import torch_dipu
import numpy as np

mat = np.random.rand(40, 30).astype(np.float32)

x1 = torch.tensor(mat, requires_grad = True)
x2 = torch.tensor(mat, requires_grad = True, device = 'dipu')

y1 = torch.sigmoid(x1)
y2 = torch.sigmoid(x2)
assert torch.allclose(y1,y2.cpu())

y1.backward(torch.ones_like(y1))
y2.backward(torch.ones_like(y2))
assert torch.allclose(x1.grad, x2.grad.cpu())
