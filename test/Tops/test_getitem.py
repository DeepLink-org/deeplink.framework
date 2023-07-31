import torch
import torch._dynamo
from dicp.vendor.TopsGraph.opset_transform import topsgraph_opset_transform
import operator
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a1, a2):
        c1  = torch.ops.aten.add(a1, a2)
        c2  = torch.ops.aten.sub(a1, a2)
        b = tuple((c1, c2))
        res = operator.getitem(b, 0)
        return res

a1, a2 = torch.rand(2, 2, 2), torch.rand(2, 2, 2)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a1, a2)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a1, a2)

print(f"Test getitem op result:{torch.allclose(r1, r2, equal_nan=True)}")
 