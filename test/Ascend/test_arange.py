import torch
import torch._dynamo as dynamo
import torch_dipu

class MyModule(torch.nn.Module):
    def forward(self, a):
        layer0 = torch.relu(a)
        layer1 = torch.ops.aten.arange.default(10).to(torch.float32)
        layer2 = torch.ops.aten.mul.Tensor(layer1, layer0)
        return layer2

a = torch.randn(10)

dynamo.reset()
ascend_model = MyModule()
compiled_model = torch.compile(ascend_model, backend="ascendgraph")
ascend_res = compiled_model(a).cpu()

dynamo.reset()
torch_model = MyModule()
compiled_model = torch.compile(torch_model, backend="inductor")
torch_res = compiled_model(a)

print(f'Tests arange result\n{torch.allclose(ascend_res, torch_res, equal_nan=True)}')
