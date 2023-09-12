import torch
import torch._dynamo as dynamo
import random

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        layer0 = torch.ops.aten.ne.Scalar(a, b)
        return layer0

a = torch.randn(5, 5)
b = random.random()

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b)

print(f'Tests ne result\n{torch.allclose(tops_res, torch_res, equal_nan=True)}')
print(a)
print(b)
print(tops_res)
print(torch_res)
