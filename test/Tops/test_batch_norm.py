import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, inputs):
        m = torch.nn.BatchNorm2d(100)
        output = m(inputs)
        return output

inputs = torch.randn(20, 100, 35, 45)
 
enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1= compiled_model(inputs)

torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(inputs)

print(f"Test batch_norm op result:{torch.allclose(r1, r2, equal_nan=True)}")
