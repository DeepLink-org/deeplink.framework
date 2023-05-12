import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b, c):
        unsq = torch.ops.aten.unsqueeze(a, b)
        sq = torch.ops.aten.squeeze(a, c)
        return unsq, sq

a1 = torch.randn(1, 3, 3)

menflame = MyModule()
print("##########################")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1, t2 = compiled_model(a1, 1, 0)
print(f'\n**************\n unsqueeze test \n {t1}\n**************\n')
print(f'\n**************\n squeeze test   \n {t2}\n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1, r2 = torchm(a1,1,  0)
print(f'\n**************\n unsqueeze ref \n {t1}\n**************\n')
print(f'\n**************\n squeeze ref   \n {t2}\n**************\n')

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.allclose(t2, r2, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')
print(f'final\n{torch.eq(t2, r2)}')