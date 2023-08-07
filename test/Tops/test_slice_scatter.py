import torch
import torch._dynamo as dynamo

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        layer0 = torch.ops.aten.slice_scatter.default(a, b)
        layer1 = torch.ops.aten.slice_scatter.default(a, b, start=6)
        layer2 = torch.ops.aten.slice_scatter.default(a, b, start=2, end=8, step=2)
        layer3 = torch.ops.aten.add(layer0, layer1)
        layer4 = torch.ops.aten.mul(layer2, layer3)
        return layer4

a = torch.randn(8, 8)
b = torch.zeros(1, 8)

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b)

print(f'Tests slice_scatter result\n{torch.allclose(tops_res, torch_res, equal_nan=True)}')
