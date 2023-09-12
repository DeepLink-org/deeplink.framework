import torch
import torch._dynamo as dynamo
import random

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        layer0 = torch.ops.aten.mul.Tensor(a, a)
        layer1 = torch.ops.prims.iota.default(5, start=b, step=b, dtype=torch.int64, device="cpu", requires_grad=False)
        layer2 = torch.ops.aten.add.Tensor(layer0, layer1)
        return layer2

a = torch.arange(5)
b = random.randint(1, 10)

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b)

print(f'Tests iota result\n{torch.allclose(tops_res, torch_res, equal_nan=True)}')
