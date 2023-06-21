import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        output = torch.ops.aten.where.self(x <= 0, x, y)
        return output

x = torch.randn(3, 2)
y = torch.ones(3, 2, dtype=torch.float32)

m = MyModule()
compiled_model = torch.compile(m, backend="topsgraph")
r1= compiled_model(x, y)
 
torch._dynamo.reset()

m = MyModule()
r2 = m(x, y)

print(f'\n****************************\n')

print(f"r1: {r1}")
print(f"r2: {r2}")

print(f"r1 - r2:\n{r1 - r2}")

print(f"nan test: r1-{torch.isnan(r1).any()}, r2-{torch.isnan(r2).any()}" )
print(f'torch.allclose:{torch.allclose(r1, r2)}')
print(f'torch.eq:{torch.eq(r1, r2).all()}')
