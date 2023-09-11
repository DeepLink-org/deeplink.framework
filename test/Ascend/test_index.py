import torch
import torch._dynamo as dynamo
import torch_dipu

def fn(a, b):
    layer0 = torch.relu(a)
    layer1 = torch.ops.aten.index.Tensor(layer0, b)
    return layer1

a = torch.randn(3, 5) + 5
b = [torch.tensor([[1, 2]])]

dynamo.reset()
compiled_model = torch.compile(fn, backend="ascendgraph")
ascend_res = compiled_model(a, b).cpu()

dynamo.reset()
compiled_model = torch.compile(fn, backend="inductor")
torch_res = compiled_model(a, b)

print(f'Tests index result\n{torch.allclose(ascend_res, torch_res, equal_nan=True)}')
