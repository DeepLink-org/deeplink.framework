import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, a, b, c, d):
        res_permute = torch.ops.aten.permute(a, b)
        res_transpose = torch.ops.aten.transpose(a, c, d)

        return res_permute, res_transpose

a = torch.randn(2, 3, 5)
b = (2, 0, 1)
c = 0
d = 1

enflame_model = MyModule()
compiled_model = torch.compile(enflame_model, backend="topsgraph")
resenflame = compiled_model(a, b, c, d)

torch_model = MyModule()
restorch = torch_model(a, b, c, d)

menflame = MyModule()
print("##########################")
print(f'\n*******result*******\n resenflame(permute) \n {resenflame[0]}\n\n resenflame(transpose) \n{resenflame[1]}\n*******result*******\n')
print(f'\n*******result*******\n restorch(permute) \n {restorch[0]}\n\n restorch(transpose) \n{restorch[1]}\n*******result*******\n')

compare_perm = torch.eq(resenflame[0], restorch[0])
compare_trans = torch.eq(resenflame[1], restorch[1])

compare_perm_1 = torch.allclose(resenflame[0], restorch[0])
compare_trans_1 = torch.allclose(resenflame[1], restorch[1])

print(f'\n*******compare result*******\n permute \n {compare_perm}\n {compare_perm_1}\n*******compare result*******\n')
print(f'\n*******compare result*******\n transpose \n {compare_trans}\n {compare_trans_1}\n*******compare result*******\n')
