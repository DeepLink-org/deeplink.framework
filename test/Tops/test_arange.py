import torch
import torch._dynamo as dynamo
import random

class MyModule(torch.nn.Module):
    def forward(self, a):
        layer0 = a + a
        layer1 = torch.ops.aten.arange.default(layer0)
        layer2 = torch.ops.aten.mul.Tensor(layer1, layer1)
        return layer2

a = random.randint(1, 10)

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a)

print(f'Tests arange result\n{torch.allclose(tops_res, torch_res, equal_nan=True)}')
