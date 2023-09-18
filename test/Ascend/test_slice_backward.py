import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        b = torch.ops.aten.slice_backward.default(a, [2, 3, 4, 5], 3, 2, 9223372036854775807, 1)
        return b

a = torch.randn(2, 3, 4, 5)
a = torch.ops.aten.slice.Tensor(a, 3, 0, 3) + 1
print(a)

ascend_model = MyModule()
compiled_model = torch.compile(ascend_model, backend="ascendgraph")
r1 = compiled_model(a).cpu()

print(r1)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a).cpu()

print(f"Test reshape op result:{torch.allclose(r1, r2, equal_nan=True)}")
