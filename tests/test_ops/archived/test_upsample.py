import torch
import torch.nn as nn
import torch_dipu

input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)

m = nn.Upsample(scale_factor=2, mode='nearest')
y1 = m(input)
y2 = m(input.cuda())
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)


m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
y1 = m(input)
y2 = m(input.cuda())
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)

m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
y1 = m(input)
y2 = m(input.cuda())
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)

# Try scaling the same data in a larger tensor
input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
input_3x3[:, :, :2, :2].copy_(input)
input_3x3

m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
# Notice that values in top left corner are the same with the small input (except at boundary)
y1 = m(input_3x3)
y2 = m(input_3x3.cuda())
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)

m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# Notice that values in top left corner are now changed
y1 = m(input_3x3)
y2 = m(input_3x3.cuda())
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)

m = nn.Upsample(scale_factor=2, mode='nearest')
x1 = input.clone()
x2 = input.cuda()
x1.requires_grad = True
x2.requires_grad = True
y1 = m(x1)
y2 = m(x2)
y1.backward(torch.ones_like(y1))
y2.backward(torch.ones_like(y2))
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)
assert torch.allclose(x1.grad, x2.grad.cpu(), atol = 1e-3)

m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
x1 = input.clone()
x2 = input.cuda()
x1.requires_grad = True
x2.requires_grad = True
y1 = m(x1)
y2 = m(x2)
y1.backward(torch.ones_like(y1))
y2.backward(torch.ones_like(y2))
assert torch.allclose(y1, y2.cpu(), atol = 1e-3)
assert torch.allclose(x1.grad, x2.grad.cpu(), atol = 1e-3)