import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform
import operator

from torch._inductor.decomposition import decompositions
del decompositions[torch.ops.aten._native_batch_norm_legit_functional.default]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        output = torch.ops.aten.where.self(x <= 0, torch.tensor(-1.5), torch.tensor(-1.5))
        return output

x = torch.randn(3, 2)


menflame = MyModule()
print("##########################")
#compiled_model = torch.compile(menflame, backend="inductor")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1= compiled_model(x)
print(f'\n**************\n test \n {t1}  \n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(x)
print(f'\n**************\n  ref \n {r1} \n**************\n')
 

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')