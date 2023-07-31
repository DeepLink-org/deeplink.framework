import torch
    
mm1 = torch.randn(2, 3)
mm2 = torch.randn(3, 3)

def fn(mm1, mm2):
    y = torch.mm(mm1, mm2)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(mm1, mm2)
print(y)
print(y.shape)