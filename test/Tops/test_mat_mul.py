import torch
import torch._dynamo
import random

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        res = torch.ops.aten.matmul(a, b)
        return res

a = torch.randn(2, 3)
b = torch.randn(3, 2)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test mat_mul op result:{torch.allclose(r1, r2, equal_nan=True)}")