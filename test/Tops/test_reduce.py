import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        r1 = torch.ops.aten.sum(a, b)
        r2 = torch.ops.aten.amax(a, b)
        return r1, r2

a1 = torch.randn(10, 10, 10)
b1 = torch.randn(10, 10)
c1 = torch.randn(10, 10)

menflame = MyModule()
#compiled_model = torch.compile(menflame, backend="inductor")
print("##########################")
compiled_model = torch.compile(menflame, backend="topsgraph")
r1, r2 = compiled_model(a1, [1, 2])
print(f'\n**************\n reducesum \n {r1}\n\n reducemax \n {r2}\n**************\n')
print("##########################")
