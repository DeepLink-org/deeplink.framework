import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        a = torch.ops.aten.expand(a, (1, 3, 3, 2))
        a = torch.ops.aten.reshape(a, (3, 3, 2))

        b = torch.ops.aten.expand(b, (1, 3, 2, 3))
        b = torch.ops.aten.reshape(b, (3, 2, 3))

        r = torch.ops.aten.bmm(a, b)
        r = torch.ops.aten.reshape(r, (1, 3, 3, 3))

        return r

a = torch.randn(1, 3, 3, 2, dtype=torch.float32)
b = torch.randn(1, 3, 2, 3, dtype=torch.float32)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1= compiled_model(a, b)

torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test bmm op result:{torch.allclose(r1, r2, equal_nan=True)}")
