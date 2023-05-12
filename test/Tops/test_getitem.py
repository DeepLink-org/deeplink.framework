import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform
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
        o1 = operator.getitem(b, 0)
        o2 = operator.getitem(b, 1)
        return o1, o2

inputs = torch.rand(3,3,3,3)
a1, a2 = torch.rand(2,2,2), torch.rand(2,2,2)

 
menflame = MyModule()
print("##########################")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1, t2 = compiled_model(a1, a2)
print(f'\n**************\n test \n {t1} {t2}\n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1, r2  = torchm(a1, a2)
print(f'\n**************\n ref \n {r1} \n{r2}\n**************\n')

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')
print(f'final\n{torch.allclose(t2, r2, equal_nan=True)}')
print(f'final\n{torch.eq(t2, r2)}')
 