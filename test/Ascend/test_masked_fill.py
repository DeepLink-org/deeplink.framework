import torch
import torch._dynamo as dynamo
import torch_dipu
import random

class MyModule(torch.nn.Module):
    def forward(self, a, b, c):
        layer0 = torch.ops.aten.add.Tensor(a, a)
        layer1 = torch.ops.aten.masked_fill.Scalar(layer0, b, c)
        layer2 = torch.ops.aten.mul.Tensor(layer0, layer1)
        return layer2

a = torch.randn(5, 5)
b = torch.tensor(random.choices([True, False]))
c = round(random.random(), 3)

dynamo.reset()
ascend_model = MyModule()
compiled_model = torch.compile(ascend_model, backend="ascendgraph")
ascend_res = compiled_model(a, b, c).cpu()

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b, c)

print(f'Tests arange result\n{torch.allclose(ascend_res, torch_res, equal_nan=True)}')
