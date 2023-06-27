import torch
import torch.fx
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, inputs):
        m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        output = m(inputs)

        return output

inputs = torch.randn(20, 16, 50, 100)


menflame = MyModule()
#compiled_model = torch.compile(menflame, backend="inductor")
print("##########################")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1 = compiled_model(inputs)
print(f'\n**************\n resenflame \n {t1}\n**************\n')

'''
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1  = torchm(inputs)
print(f'\n**************\n unsqueeze ref \n {r1}\n**************\n')

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')
'''