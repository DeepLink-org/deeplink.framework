import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.add(a, b)
        layer1 = torch.ops.aten.abs(layer0)
        layer2 = torch.ops.aten.log(layer1)
        layer3 = torch.ops.aten.div(layer2, layer0)
        return layer3

a = torch.randn(10, 10)
b = torch.randn(10, 10)

enflame_model = MyModule()

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a, b)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a, b)

print(f"Test log op result:{torch.allclose(r1, r2, equal_nan=True)}")
