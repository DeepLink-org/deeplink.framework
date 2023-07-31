import torch
    
input = torch.randn(3, 3, 4)
mat2 = torch.randn(3, 4, 5)

def fn(input, mat2):
    y = torch.bmm(input, mat2)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')

y = opt_model(input, mat2)
print(y)
print(y.shape)
print('Resources released successfully.')
