import torch
import torch.fx
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        #t0 = torch.ops.aten.sub(b, c)
        #t1 = torch.ops.aten.add(b, c)
        #t2 = torch.ops.aten.mul(b, c)
        #r1 = torch.ops.aten.addmm(t0, t1, t2)
        r1 = torch.ops.aten.le.Scalar(a, b)
        return r1

a1 = torch.randn(10, 10)
b1 = torch.randn(10, 10)
c1 = torch.randn(10, 10)

menflame = MyModule()
print("##########################")
compiled_model = torch.compile(menflame, backend="topsgraph")
resenflame = compiled_model(a1, .10)
print(f'\n**************\n test \n {resenflame}\n**************\n')

torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
rt = torchm(a1, 0.10)
print(f'\n**************\n ref \n {rt}\n**************\n')

print(f'final\n{torch.allclose(rt, resenflame, equal_nan=True)}')
print(f'final\n{torch.eq(rt, resenflame)}')