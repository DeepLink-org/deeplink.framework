import os
import torch
import torch._dynamo
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
device_id = os.getenv('DICP_TOPS_DEVICE_ID', default='0')
x = torch.arange(2, 18).reshape(4, 4)

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
r1 = compiled_model(x.to(f"dipu:{1}")).cpu()
 
torch._dynamo.reset()

torch_model = MyModule()
r2 = torch_model(x).cpu()

print(f"Test mul_dipu op result:{torch.allclose(r1, r2, equal_nan=True)}")
