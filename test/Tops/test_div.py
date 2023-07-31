import math

import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        b = math.sqrt(b)
        r = torch.ops.aten.div(a, b)
        return r

a = torch.arange(16).reshape(4, 4).float()
b = 128

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b)

torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test div op result:{torch.allclose(r1, r2, equal_nan=True)}")
