import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.mul(a, b)
        layer1 = torch.ops.aten.where(a <= 0, a, b)
        layer2 = torch.ops.aten.add(layer0, layer1)
        return layer2

a = torch.randn(5, 5)
b = torch.randn(5, 5)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(a, b)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(a, b)

print(f'Tests where result\n{torch.allclose(t1, r1, equal_nan=True)}')
