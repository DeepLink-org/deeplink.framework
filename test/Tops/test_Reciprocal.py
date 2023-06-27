import torch
import torch.fx
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform
import operator

from torch._inductor.decomposition import decompositions
del decompositions[torch.ops.aten._native_batch_norm_legit_functional.default]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, inputs):
        m =torch.ops.aten.reciprocal(inputs)
        return m

inputs = torch.randn(3,3,3)

 
menflame = MyModule()
print("##########################")
#compiled_model = torch.compile(menflame, backend="inductor")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1= compiled_model(inputs)
print(f'\n**************\n test \n {t1}  \n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(inputs)
print(f'\n**************\n  ref \n {r1} \n**************\n')
 

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')