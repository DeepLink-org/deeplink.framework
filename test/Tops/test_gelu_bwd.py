import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.mul(a, b)
        layer1 = torch.ops.aten.gelu_backward(layer0, a, approximate='none')
        layer2 = torch.ops.aten.gelu_backward(layer1, b)
        layer3 = torch.ops.aten.gelu_backward(layer1, layer2, approximate='tanh')
        layer4 = torch.ops.aten.mul(layer2, layer3)
        return layer4

a = torch.randn(5, 5)
b = torch.randn(5, 5)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(a, b)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(a, b)

print(f'Tests gelu_backward result\n{torch.allclose(t1, r1, equal_nan=True)}')
