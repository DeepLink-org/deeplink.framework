import torch
import torch._dynamo
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b, c, d):
        res_permute = torch.ops.aten.permute(a, b)
        res_transpose = torch.ops.aten.transpose(a, c, d)

        return res_permute, res_transpose

a = torch.randn(2, 3, 5)
b = (2, 0, 1)
c = 0
d = 1

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1, r2 = compiled_model(a, b, c, d)
 
torch._dynamo.reset()

torch_model = MyModule()
r3, r4 = torch_model(a, b, c, d)

print(f"Test permute op result:{torch.allclose(r1, r3, equal_nan=True)}")
print(f"Test transpose op result:{torch.allclose(r2, r4, equal_nan=True)}")