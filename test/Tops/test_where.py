import torch
import torch.fx
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform
import operator

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        output = torch.ops.aten.where.self(x <= 0, torch.tensor(-1.5), torch.tensor(-1.5))
        return output

x = torch.randn(3, 2)

m = MyModule()
compiled_model = torch.compile(m, backend="topsgraph")
r1= compiled_model(x)
 
torch._dynamo.reset()

m = MyModule()
compiled_model = torch.compile(m, backend="inductor")
r2 = compiled_model(x)

print(f'\n****************************\n')

print(r1)
print(r2)

print(f"r1 - r2:\n{r1 - r2}")
print(f"nan test: r1-{torch.isnan(r1).any()}, r2-{torch.isnan(r2).any()}" )
print(f'torch.allclose:\n{torch.allclose(r1, r2, equal_nan=True)}')
print(f'torch.eq:{torch.eq(r1, r2).all()}')
