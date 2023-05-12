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
        c1 = torch.ops.aten.view_as_complex.default(x)
        m1 = torch.ops.aten.view_as_real.default(c1)
        return c1, m1

x = torch.randn(1, 12, 32, 64, 2)

menflame = MyModule()

compiled_model = torch.compile(menflame, backend="topsgraph")
comx1, real1 = compiled_model(x)

torch._dynamo.reset()

refcomx1 = torch.ops.aten.view_as_complex.default(x)
refreal1 = torch.ops.aten.view_as_real.default(refcomx1)

print(f'Test Real and Complex Result for view_as_complex: {torch.allclose(comx1, refcomx1, equal_nan=True)}')
print(f'Test Real and Complex Result for view_as_real: {torch.allclose(real1, refreal1, equal_nan=True)}')