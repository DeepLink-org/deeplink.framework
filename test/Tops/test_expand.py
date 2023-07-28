import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b, c):
        layer0 = torch.ops.aten.add(a, a)
        layer1 = torch.ops.aten.expand(layer0, (b, c))
        layer2 = torch.ops.aten.add(layer1, layer1)
        return layer2

a = torch.tensor([[1], [2], [3]])
b = 3
c = 4

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b, c)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b, c)

print(f"Test expand op result:{torch.allclose(r1, r2, equal_nan=True)}")
