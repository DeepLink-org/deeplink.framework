import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, input):
        value0 = torch.ops.aten.mul(input, input)
        value1 = torch.nn.functional.gelu(value0, approximate='none')
        value2 = torch.nn.functional.gelu(value1)
        value3 = torch.nn.functional.gelu(value2, approximate='tanh')
        res = torch.ops.aten.mul(value3, value3)
        return res

x = torch.randn(5, 5)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(x)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(x)

print(f'Tests Gelu Result\n{torch.allclose(t1, r1, equal_nan=True)}')
