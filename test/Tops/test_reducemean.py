import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, inputs):
        output = torch.ops.aten.mean.dim(inputs, [2, 3], True)

        return output

inputs = torch.rand(3,3,3,3)


menflame = MyModule()
print("##########################")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(inputs)
print(f'\n**************\n test \n {t1}\n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1  = torchm(inputs)
print(f'\n**************\n  ref \n {r1}\n**************\n')

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')
 