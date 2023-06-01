import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, input):
        input = torch.ops.aten.add(input, input)
        res = torch.nn.functional.hardswish(input)
        res = torch.ops.aten.mul(res, res)
        return res

x = torch.randn(10, 10)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(x)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(x)

print(f'Tests Hardswish Result\n{torch.allclose(t1, r1, equal_nan=True)}')
