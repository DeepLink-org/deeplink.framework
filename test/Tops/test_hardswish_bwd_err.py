import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        layer0 = torch.ops.aten.add(a, b)
        layer1 = torch.ops.aten.mul(a, b)
        layer2 = torch.ops.aten.add(layer0, layer1)
        layer3 = torch.nn.functional.hardswish(layer0)
        layer4 = torch.ops.aten.mul(layer3, layer3)
        loss = torch.nn.CrossEntropyLoss()
        result_loss = loss(layer4, b)
        result_loss.backward()
        return result_loss

a = torch.randn(10, 10, requires_grad=True)
b = torch.randn(10, 10)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(a, b)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(a, b)

print(f'Tests hardswish_backward result\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'Please check if generated code contains \'HardSwishGrad\'')
