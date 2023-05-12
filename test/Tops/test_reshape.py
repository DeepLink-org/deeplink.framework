import torch
import torch.fx
from third_party.DICP.TopsGraph.opset_transform import topsgraph_opset_transform
import operator

from torch._inductor.decomposition import decompositions
del decompositions[torch.ops.aten._native_batch_norm_legit_functional.default]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, input):
        output = input.view(1, 3, 2, 4)
        return output

input = torch.randn(1, 2, 3, 4)

 
menflame = MyModule()
print("##########################")
#compiled_model = torch.compile(menflame, backend="inductor")
compiled_model = torch.compile(menflame, backend="topsgraph")
t1= compiled_model(input)
print(f'\n**************\n test \n {t1}  \n**************\n')
 
torch._dynamo.reset()
tm = MyModule()
torchm = torch.compile(tm)
r1 = torchm(input)
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