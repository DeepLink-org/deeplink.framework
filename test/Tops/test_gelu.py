import torch
import torch._dynamo

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

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(x)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(x)

print(f"Test gelu op result:{torch.allclose(r1, r2, equal_nan=True)}")