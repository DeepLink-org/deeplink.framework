import torch
import torch._dynamo as dynamo
import random

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        layer0 = torch.ops.aten.add(a, a)
        layer1 = torch.ops.aten.div.Tensor(layer0, a)
        layer2 = torch.ops.aten.div.Tensor(layer1, b)
        layer3 = torch.ops.aten.div.Scalar(layer1, b)
        layer4 = torch.ops.aten.mul(layer2, layer3)
        return layer4

a = torch.randn(5, 5)
b = random.random()

tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b)

dynamo.reset()
torch_model = MyModule()
r2 = torch_model(a, b)

print(f'Tests div result\n{torch.allclose(tops_res, torch_res, equal_nan=True)}')
