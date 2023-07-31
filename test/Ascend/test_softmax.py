import torch

softmax = torch.randn(2, 3)

def fn(x):
    y = torch.nn.functional.softmax(x, -1)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(softmax)
print(y)
print(y.shape)
print('Resources released successfully.')
