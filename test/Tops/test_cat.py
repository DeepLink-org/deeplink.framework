import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a):
        output = torch.cat((a, a), 0)
        return output

a = torch.arange(4, dtype=torch.float32).reshape(2, 2)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a)

print(f"Test cat op result:{torch.allclose(r1, r2, equal_nan=True)}")
