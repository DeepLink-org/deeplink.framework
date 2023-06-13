import torch
import torch.fx
import random

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.zeros((a, b))
        layer1 = torch.ops.aten.ones((a, b))
        layer2 = torch.ops.aten.full((a, b), float(a))
        layer3 = torch.ops.aten.add(layer0, layer1)
        layer4 = torch.ops.aten.add(layer2, layer3)
        return layer4

a = random.randint(1, 10)
b = random.randint(1, 10)
c = torch.randn(a, b)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
res_tops = compiled_model(a, b)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
res_torch = torchm(a, b)

print(f'Tests zeros, ones, full result\n{torch.allclose(res_tops, res_torch, equal_nan=True)}')
