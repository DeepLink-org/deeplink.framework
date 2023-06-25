import torch
import torch_dipu
import torch.nn as nn

# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50, 100)
input_d = input.cuda()
input_d.requires_grad = True
output = m(input)

m_d = m.cuda()
output_d = m_d(input_d)
output_d.backward(torch.ones_like(output_d))

# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)
output = m(input)

# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
h.size()
output = upsample(h, output_size=input.size())
output.size()