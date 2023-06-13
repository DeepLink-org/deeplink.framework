import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.add(a, b)
        layer1 = torch.ops.prims.convert_element_type(layer0, torch.uint8)
        layer2 = torch.ops.aten.mul(layer1, layer1)
        return layer2

a = torch.randn(10, 10, dtype=torch.double)
b = torch.randn(10, 10, dtype=torch.float)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
resenflame = compiled_model(a, b)

torch_model = MyModule()
torch_model = torch.compile(torch_model)
restorch = torch_model(a, b)

compare = torch.allclose(resenflame, restorch, equal_nan=True)

print(f'Tests convert reslut\n{compare}')
