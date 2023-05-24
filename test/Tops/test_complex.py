import torch
import torch.fx
import operator

from torch._inductor.decomposition import decompositions
del decompositions[torch.ops.aten._native_batch_norm_legit_functional.default]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        output = torch.view_as_complex(x)
        return output

x = torch.randn(5, 2)

menflame = MyModule()
compiled_model = torch.compile(menflame, backend="topsgraph")
t1= compiled_model(x)
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(x)

print(f'Tets view_as_omplex Result\n{torch.allclose(t1, r1, equal_nan=True)}')