import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b, c):
        unsq = torch.ops.aten.unsqueeze(a, b)
        sq = torch.ops.aten.squeeze(a, c)
        return unsq, sq

a = torch.randn(1, 3, 3)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1, r2 = compiled_model(a, 1, 0)
 
torch._dynamo.reset()

torch_model = MyModule()
r3, r4 = torch_model(a, 1, 0)

print(f"Test unsqueeze op result:{torch.allclose(r1, r3, equal_nan=True)}")
print(f"Test squeeze op result:{torch.allclose(r2, r4, equal_nan=True)}")
