import torch
import torch._dynamo

from torch._inductor.decomposition import decompositions
del decompositions[torch.ops.aten._native_batch_norm_legit_functional.default]
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, inputs):
        m = torch.nn.LogSoftmax(dim=1)
        output = m(inputs)
        return output

x = torch.randn(2, 3)

enflame_model = MyModule()

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(x)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(x)

print(f"Test log_softmax op result:{torch.allclose(r1, r2, equal_nan=True)}")