import torch
import torch.fx
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a):
        a = torch.ops.aten.abs(a)
        res = torch.ops.aten.rsqrt(a)
        res = torch.ops.aten.mul(res, res)
        return res

a = torch.rand(10, 10)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
resenflame = compiled_model(a)

torch_model = MyModule()
restorch = torch_model(a)

print("##########################")
print(f'\n*******result*******\n {resenflame} \n*******result*******\n')
print(f'\n*******result*******\n {restorch}   \n*******result*******\n')

compare = torch.allclose(resenflame, restorch)

print(f'\n*******compare result*******\n  {compare} \n*******compare result*******\n')
