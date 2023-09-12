import torch
import torch._dynamo as dynamo

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        layer0 = torch.ops.aten.new_empty_strided.default(a, (b, b), (b, 1))
        return layer0

a = torch.randn(5, 5)
b = 5

dynamo.reset()
tops_model = MyModule()
compiled_model = torch.compile(tops_model, backend="topsgraph")
tops_res = compiled_model(a, b)

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a, b)

print(tops_res)
print(torch_res)
