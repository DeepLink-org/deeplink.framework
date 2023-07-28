import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b, c):
        t0 = torch.ops.aten.sub(b, c)
        t1 = torch.ops.aten.add(b, c)
        t2 = torch.ops.aten.mul(b, c)
        r1 = torch.ops.aten.addmm(t0, t1, t2)
        return r1

a = torch.randn(10, 10)
b = torch.randn(10, 10)
c = torch.randn(10, 10)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b, c)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b, c)

print(f"Test trans op result:{torch.allclose(r1, r2, equal_nan=True)}")
