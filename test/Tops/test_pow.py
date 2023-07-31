import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        a = torch.ops.aten.add(a, a)
        res = torch.ops.aten.pow(a, b)
        res = torch.ops.aten.mul(res, res)
        return res

a = torch.randn(10, 10)
b = 3

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test pow op result:{torch.allclose(r1, r2, equal_nan=True)}")
