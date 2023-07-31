import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        r1 = torch.ops.aten.sum(a, b)
        r2 = torch.ops.aten.amax(a, b)
        return r1, r2

a = torch.randn(10, 10, 10)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1, r2 = compiled_model(a, [1, 2])
 
torch._dynamo.reset()

torch_model = MyModule()
r3, r4 = torch_model(a, [1, 2])

print(f"Test reduce_sum op result:{torch.allclose(r1, r3, equal_nan=True)}")
print(f"Test reduce_amax op result:{torch.allclose(r2, r4, equal_nan=True)}")
