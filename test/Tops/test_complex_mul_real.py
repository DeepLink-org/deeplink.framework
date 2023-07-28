import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        c1 = torch.ops.aten.view_as_complex.default(x)
        c2 = torch.ops.aten.view_as_complex.default(y)
        m1 = torch.ops.aten.mul.Tensor(c1, c2)
        res = torch.ops.aten.view_as_real.default(m1)
        return res

x = torch.randn(1, 12, 32, 64, 2)
y = torch.randn(1, 12, 32, 64, 2)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(x, y)

torch._dynamo.reset()

rc1 = torch.ops.aten.view_as_complex.default(x)
rc2 = torch.ops.aten.view_as_complex.default(y)
rm = torch.ops.aten.mul.Tensor(rc1, rc2)
r2 = torch.ops.aten.view_as_real.default(rm)

print(f"Test complex_mul_real op result:{torch.allclose(r1, r2, equal_nan=True)}")
