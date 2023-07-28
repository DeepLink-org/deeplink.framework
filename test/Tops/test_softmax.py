import torch
import torch._dynamo
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        m = torch.nn.Softmax(dim=3)
        output = m(input)
        return output

a = torch.randn(1, 32, 32, 32, dtype=torch.float32)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(a)
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(a)

print(f"Test softmax op result:{torch.allclose(r1, r2, equal_nan=True)}")
