import torch
import torch.fx
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform
import operator
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, t, d,  c):
        o = torch.ops.aten.gather(t, d, c)
        return o

t = torch.tensor([[1, 2], [3, 4]])
c = torch.tensor([[0, 0], [1, 0]])
d = 1
 
menflame = MyModule()
print("##########################")
#compiled_model = torch.compile(menflame, backend="inductor")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1= compiled_model(t,d, c)
print(f'\n**************\n test \n {t1}  \n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(t,d, c)
print(f'\n**************\n  ref \n {r1} \n**************\n')
 

print(f'final\n{torch.allclose(t1, r1, equal_nan=True)}')
print(f'final\n{torch.eq(t1, r1)}')
'''
    std::vector<int64_t> op0_in_shape{2, 2};;
    builder::Type op0_input_type(op0_in_shape, builder::PrimitiveType::S64());;
    builder::Op op0 = hlir_builder->CreateInput(op0_input_type);;
    std::vector<int64_t> op1_in_shape{2, 2};;
    builder::Type op1_input_type(op1_in_shape, builder::PrimitiveType::S64());;
    builder::Op op1 = hlir_builder->CreateInput(op1_input_type);;
    int op2_value0 = 1;;
    std::vector<int64_t> op2_const_in_shape0{1};;
    builder::Type op2_value_type0(op2_const_in_shape0, ptype);;
    builder::Op op2_const0 = builder::Const(hlir_builder, static_cast<void *>(&op2_value0), op2_value_type0);;
    builder::Type gather_typeop2({2, 2}, builder::PrimitiveType::F32());;
      auto op2 = builder::Gather(hlir_builder, op0, op1, op2_const0, gather_typeop2);;
      '''