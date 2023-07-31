import torch
import torch._dynamo
from dicp.vendor.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        r = torch.ops.aten.sub(a, b)
        return r

a = torch.randn(1, 32, 32, 32, dtype=torch.float16)
b = torch.randn(1, 32, 32, 1, dtype=torch.float16)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test sub op result:{torch.allclose(r1, r2, equal_nan=True)}")