import torch
import torch.fx
import random

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        layer0 = torch.ops.aten.add(x, x)
        layer1 = torch.ops.aten.abs(layer0)
        layer2 = torch.ops.aten.mm(layer1, y)
        layer3 = torch.ops.aten.abs(layer2)
        return layer3

a = random.randint(1, 10)
b = random.randint(1, 10)
c = random.randint(1, 10)
x = torch.randn(a, b)
y = torch.randn(b, c)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
resenflame = compiled_model(x, y)

torch_model = MyModule()
restorch = torch_model(x, y)

compare = torch.allclose(resenflame, restorch, equal_nan=True)

print(f'Tests mm reslut\n{compare}')
