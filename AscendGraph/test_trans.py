import torch
from opset_convert import ascendgraph_opset_convert

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        a = torch.add(x, y)
        z = torch.add(x, a)
        a = torch.abs(z)
        return tuple([a])

m = MyModule()
traced = torch.fx.symbolic_trace(m)
print(traced.graph)
transformed = ascendgraph_opset_convert(traced)
print(transformed.graph)