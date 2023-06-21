import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        c1 = torch.ops.aten.view_as_complex.default(x)
        m1 = torch.ops.aten.view_as_real.default(c1)
        return m1

x = torch.randn(1, 12, 32, 64, 2)

m = MyModule()
compiled_model = torch.compile(m, backend="topsgraph")
r1 = compiled_model(x)

torch._dynamo.reset()

t = torch.ops.aten.view_as_complex.default(x)
r2 = torch.ops.aten.view_as_real.default(t)

print(f'\n****************************\n')

print(f"r1: {r1}")
print(f"r2: {r2}")

print(f"r1 - r2:\n{r1 - r2}")

print(f"nan test: r1-{torch.isnan(r1).any()}, r2-{torch.isnan(r2).any()}" )
print(f'torch.allclose:{torch.allclose(r1, r2)}')
print(f'torch.eq:{torch.eq(r1, r2).all()}')
