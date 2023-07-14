import os
import torch
import torch.fx
import torch_dipu


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        output = torch.mul(x, 2)
        return output

os.environ['DICP_TOPS_DIPU'] = 'True'
os.environ['DIPU_MOCK_CUDA'] = 'false'
device_id = os.getenv('DICP_TOPS_DEVICE_ID', default='0')
x = torch.arange(2, 18).reshape(4, 4)

m = MyModule()
compiled_model = torch.compile(m, backend="topsgraph")
r1= compiled_model(x.to(f"dipu:{device_id}")).cpu()

torch._dynamo.reset()

m = MyModule()
r2 = m(x).cpu()

print(f'\n****************************\n')

print(f"r1: {r1}", flush=True)
print(f"r2: {r2}", flush=True)

print(f"r1 - r2:\n{r1 - r2}")

print(f"nan test: r1-{torch.isnan(r1).any()}, r2-{torch.isnan(r2).any()}" )
print(f'torch.allclose:{torch.allclose(r1, r2)}')
print(f'torch.eq:{torch.eq(r1, r2).all()}')
