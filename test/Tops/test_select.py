import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a):
        r = torch.ops.aten.select(a, 0, -1)
        r = r + 1
        return r

a = torch.arange(16, dtype=torch.float32).reshape(4, 4)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a)

print(f"Test select op result:{torch.allclose(r1, r2, equal_nan=True)}")
