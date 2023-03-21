import torch
import torch.fx
from opset_transform import topsgraph_opset_transform
from torch.tops.operator import *

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b):
        a = torch.ops.aten.rsqrt.default(a, b)
        return a
        #return self.linear(x + self.param)#.clamp(min=0.0, max=1.0)


m = MyModule()
traced = torch.fx.symbolic_trace(m)
print(traced.graph)
print("do transforma")
transformed = topsgraph_opset_transform(traced)

print(transformed.graph)