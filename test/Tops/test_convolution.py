import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)
        
    def forward(self, inputs, weights, bias):
        m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(0, 0), dilation=(1, 1), bias=False)
        m.weights = weights
        # m.bias = bias 
        output = m(inputs)
        return output

inputs = torch.randn(20, 16, 50, 100)
weights = torch.randn(33, 16, 3, 5)
bias = torch.randn(33)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(inputs, weights, bias)

torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(inputs, weights, bias)

print(f"Test convolution op result:{torch.allclose(r1, r2, equal_nan=True)}")
