import torch
    
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(3, 2)


def fn(a, b, c):
    t1 = torch.add(a, b)
    t2 = torch.add(a, t1, alpha=1.51)
    t3 = torch.add(t2, 8.9)
    t4 = torch.add(t3, 9.9, alpha=1.61)
    x = torch.sub(t4, b)
    y = torch.mm(x, c)
    return torch.sub(y, 2.12)
    

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(a, b, c)
y2 = fn(a, b, c)
print(y)
print(y2)
assert torch.allclose(y.to('cpu'), y2)