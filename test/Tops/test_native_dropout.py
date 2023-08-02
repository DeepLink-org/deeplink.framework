import torch
import torch._dynamo as dynamo
import random

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        layer0 = torch.ops.aten.add.Tensor(a, a)
        layer1 = torch.ops.aten.native_dropout.default(layer0, b, True)
        return layer1

a = torch.randn(5, 5)
b = round(random.random(), 3)

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b)
