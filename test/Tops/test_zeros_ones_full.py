import random

import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.zeros((a, b))
        layer1 = torch.ops.aten.ones((a, b))
        layer2 = torch.ops.aten.full((a, b), float(a))
        layer3 = torch.ops.aten.add(layer0, layer1)
        layer4 = torch.ops.aten.add(layer2, layer3)
        return layer4

a = random.randint(1, 10)
b = random.randint(1, 10)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test zeros_ones_full op result:{torch.allclose(r1, r2, equal_nan=True)}")
