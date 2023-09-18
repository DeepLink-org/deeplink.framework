import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        res = torch.ops.aten.rsqrt(a)
        return res

a = torch.randn(10)
print(a)

ascend_model = MyModule()
compiled_model = torch.compile(ascend_model, backend="ascendgraph")
r1 = compiled_model(a)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a)

print(f"Test reshape op result:{torch.allclose(r1, r2, equal_nan=True)}")
