

op_set = {"Lt":"Less", 
                       "Le":"LessEqual",
                       "Gt":"Greater",
                       "Ge":"GreaterEqual",
                       "Eq":"Equal",
                       "Ne":"NotEqual",a
                       "Sum":"ReduceSum",
                       "Max":"ReduceMax",
                       "Amax":"ReduceMax",
                       "Min":"ReduceMin",
                       "Mean":"ReduceMean",
                       "View":"Reshape",
                       "Mm":"Gemm",
                       "Where":"Select",
                       "T":"Transpose",
                       "Permute":"Transpose",
                       "_Log_Softmax":"Softmax",
                       "Nll_Loss_Forward":"CTCLoss",
                       "Convert_Element_Type":"Convert"}


fixed_code = '''from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

kernel_cpp_0 = async_compile.enflame(\'''
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dtu/hlir_builder/hlir_builder.h"
#include "dtu/hlir_builder/hlir_builder_client_ops.h"

#include "common/dtu_utils.h"
#include "common/dtu_utils.cpp"
#include "enflame/conv2d_grad.cpp"
#include "enflame/max_pool2d_grad.cpp"

topsExecutable_t exe_ptr;

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto ptype = builder::PrimitiveType::F32();
  
{self.src_code}
{res_str}
  return hlir_builder;
}

extern "C" void compile(void){
  // stage 1: build the ir
  auto hlir_builder = build_sample();

  // stage 2: compile
  std::cout << "ccccc " << std::endl;
  compile(hlir_builder, &exe_ptr);
}

extern "C" void run({input_paras}
                    {output_paras}) {
  // stage 3: input and output
  std::vector<void *> input_ptrs;
{inputs}
  std::vector<void *> output_ptrs;
{outputs}
  // stage 4: run
  run(exe_ptr, input_ptrs, output_ptrs);
}
\''')\n

async_compile.wait(globals())
del async_compile

{kernel_cal}

'''

if __name__ == "__main__":
    print(fixed_code)