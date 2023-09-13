import torch
import torch_dipu

input = torch.randn(3, 2, requires_grad=True)
target = torch.rand(3, 2, requires_grad=False)
loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(input), target)
loss.backward()

input1 = input.detach().clone().cuda()
input1.requires_grad = True
target1 = target.cuda()
loss1 = torch.nn.functional.binary_cross_entropy(torch.sigmoid(input1), target1)
loss1.backward()

assert torch.allclose(loss1.cpu(), loss, atol = 1e-3, rtol = 1e-3)
assert torch.allclose(input1.grad.cpu(), input.grad, atol = 1e-3, rtol = 1e-3)


input = torch.randn(3, 2, requires_grad=True)
target = torch.rand(3, 2, requires_grad=False)
loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(input), target, reduction = 'none')
loss.backward(torch.ones_like(input))

input1 = input.detach().clone().cuda()
input1.requires_grad = True
target1 = target.cuda()
loss1 = torch.nn.functional.binary_cross_entropy(torch.sigmoid(input1), target1, reduction = 'none')
loss1.backward(torch.ones_like(input1))

assert torch.allclose(loss1.cpu(), loss, atol = 1e-3, rtol = 1e-3)
assert torch.allclose(input1.grad.cpu(), input.grad, atol = 1e-3, rtol = 1e-3)