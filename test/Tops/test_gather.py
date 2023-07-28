import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, t, d,  c):
        o = torch.ops.aten.gather(t, d, c)
        return o

t = torch.tensor([[1, 2], [3, 4]])
c = torch.tensor([[0, 0], [1, 0]])
d = 1
 
enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1= compiled_model(t,d, c)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(t,d, c)

print(f"Test gather op result:{torch.allclose(r1, r2, equal_nan=True)}")
