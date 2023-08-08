import torch
import torch._dynamo as dynamo
import random

class MyModule(torch.nn.Module):
    def forward(self, a, b, c):
        layer0 = torch.tensor(torch.finfo(a.dtype).min)
        layer1 = torch.tensor([b, c])
        return layer0, layer1

a = torch.randn(5, 5, device="cpu")
b = round(random.random(), 3)
c = round(random.random(), 3)

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b, c)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b, c)

print(f'Tests lift_fresh_copy result\n{torch.allclose(tops_res[0], torch_res[0], equal_nan=True) and torch.allclose(tops_res[1], torch_res[1], equal_nan=True)}')
