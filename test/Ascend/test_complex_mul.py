import torch

a = torch.randn([4, 2], dtype=torch.float32)
b = torch.randn([4, 2], dtype=torch.float32)

def fn (a, b):
    ct1 = torch.view_as_complex(a)
    ct2 = torch.view_as_complex(b)
    res =  ct1 * ct2
    return torch.view_as_real(res)

opt_model = torch.compile(fn, backend='ascendgraph')

y = opt_model(a, b)
y2 = fn(a, b)
print(y)
print(y2)
assert torch.allclose(y.to('cpu'), y2)
