import random

import torch
import torch._dynamo

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
r1 = compiled_model(x, y)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(x, y)

print(f"Test mm op result:{torch.allclose(r1, r2, equal_nan=True)}")
