import torch

a = torch.randn(1, 2, 3, 4)

def fn(x):
    y = torch.transpose(x, 1, 2).contiguous()
    return y

#opt_model = torch.compile(fn, backend='ascendgraph')
opt_model = torch.compile(fn, backend='inductor')

y = opt_model(a)
print(y)
print(y.shape)
print('Resources released successfully.')
