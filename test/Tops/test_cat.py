import torch
import torch.fx

from torch._inductor.decomposition import decompositions
del decompositions[torch.ops.aten._native_batch_norm_legit_functional.default]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        output1 = torch.cat((x, y), -2)
        output2 = torch.cat((x, y), 3)
        return output1, output2

x = torch.randn(1, 2, 32, 64)
y = torch.randn(1, 2, 32, 64)


menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1, t2= compiled_model(x, y)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1, r2 = torchm(x, y)

print(f'Tets cat Result\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'Tets cat Result\n{torch.allclose(t2, r2, equal_nan=True)}')
