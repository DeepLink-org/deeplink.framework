import contextlib
import dataclasses
import functools
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import sympy

import torch

from torch._prims_common import is_float_dtype
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch._inductor.sizevars import SimplifyIndexing
import torch.fx.traceback as fx_traceback

from torch._inductor import codecache, config, ir, metrics
from torch._inductor.utils import sympy_product, sympy_subs, sympy_symbol
from torch._inductor.virtualized import ops, V
from .common import (
    BracesBuffer,
    CppWrapperKernelArgs,
    DeferredIndentedBuffer,
    ExprPrinter,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
)


DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}

DTYPE_TO_ATEN = {
    torch.float32: "at::ScalarType::Float",
    torch.float64: "at::ScalarType::Double",
    torch.float16: "at::ScalarType::Half",
    torch.int64: "at::ScalarType::Long",
    torch.int32: "at::ScalarType::Int",
    torch.int16: "at::ScalarType::Short",
    torch.int8: "at::ScalarType::Char",
    torch.uint8: "at::ScalarType::Byte",
    torch.bool: "at::ScalarType::Bool",
    torch.bfloat16: "at::ScalarType::BFloat16",
}

INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}

type_set = {"torch.float32": "builder::PrimitiveType::F32()",
            "torch.int64": "builder::PrimitiveType::S64()"}

op_set = {"Lt":"Less", 
                       "Le":"LessEqual",
                       "Gt":"Greater",
                       "Ge":"GreaterEqual",
                       "Eq":"Equal",
                       "Ne":"NotEqual",
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

class EnflameCodegen(torch.fx.Interpreter):
    def __init__(self, graph):
        self.header = IndentedBuffer()
        self.code = IndentedBuffer()
        self.compile = IndentedBuffer()
        self.run = IndentedBuffer()
        self.call = IndentedBuffer()
        self.src_code = ''
        
        self.args_dict = {}
        self.input_args =[]
        self.output_args = []
        
        self.testing = IndentedBuffer()

        self.graph = graph
        super().__init__(graph)
        self.override = EnflameOverrides

    def placeholder(self, name, target, args, kwargs):    
        self.args_dict[name] = 'tmp' + str(len(self.args_dict))
        self.input_args.append(self.cur_node)
        
        in_shape = self.get_shape()
        self.src_code += f'  std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape};\n'

        data_type = self.cur_node.meta['val'].dtype.__str__()
        self.src_code += f'  builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, {type_set[data_type]});\n'
            
        self.src_code += f'  builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);\n\n'

    def call_function(self, name, target, args, kwargs):   
        if name not in self.args_dict.keys():
            self.args_dict[name] = 'tmp' + str(len(self.args_dict))
        
        operation = self.get_op_name(target)
        
        tmp_code = getattr(self.override, self.get_op_name(target))(self.args_dict, self.cur_node, args)
        self.src_code += tmp_code
        
        return
    
    def get_op_name(self, target):
        if target.__name__.split('::')[-1].title().split('.')[0] in op_set.keys():
            operation = op_set[target.__name__.split('::')[-1].title().split('.')[0]]
        else:
            operation = target.__name__.split('::')[-1].title().split('.')[0]
            
        operation = operation.split('_')[0]
        
        return operation

    def output(self, name, target, args, kwargs):
        for i in range(0, len(args[0])):
            self.output_args.append(args[0][i])   

    def run_node(self, n : Node) -> Any:
        self.cur_node = n
        op = n.op
        name = n.name
        target = n.target
        args = n.args
        kwargs = n.kwargs

        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        return getattr(self, op)(name, target, args, kwargs) 
    
    def codegen(self):
        self.run()
        
        tmp_str = []
        res_str = ''
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                continue
            else:
                tmp_str.append(self.args_dict[self.output_args[i].name])

        self.src_code += f'  hlir_builder->SetOutput({"{" + ", ".join(tmp_str) + "}"});\n' 

        self.code.splice(
            f"""
                topsExecutable_t exe_ptr;
                std::shared_ptr<builder::Builder> build_sample() {'{'}
                auto hlir_builder = std::make_shared<builder::Builder>();
                hlir_builder->SetShapeInference(true);
                auto ptype = builder::PrimitiveType::F32();

                {self.src_code}
                
                return hlir_builder;{'}'}
            """
        )
        
        test_code = self.generate_header() + self.code.getvalue() + self.generate_compile() + self.generate_run() + self.generate_cal()
        print(test_code)
        raise ValueError("test")
        return self.generate_header() + self.code.getvalue() + self.generate_compile() + self.generate_run() + self.generate_cal()
    
    def get_shape(self):
        shape = '{' + str(self.cur_node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        return

    def generate_header():
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
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
            """
        )
        
        return self.header.getvalue()

    def generate_compile():
        self.compile.splice(
            f"""
                extern "C" void compile(void){'{'}
                    // stage 1: build the ir
                    auto hlir_builder = build_sample();

                    // stage 2: compile
                    std::cout << "ccccc " << std::endl;
                    compile(hlir_builder, &exe_ptr);
                {'}'}
            """
        )
        
        return self.compile.getvalue()
    
    def generate_run():
        input_paras = ''
        for i in range(0, len(self.input_args)):
            input_paras += f'float* input_ptr{str(i)}, '
        
        output_paras = []
        for i in range(0, len(self.output_args)):
            output_paras.append(f'float* output_ptr{str(i)}')
        output_paras = ', '.join(output_paras)
        
        inputs = ''
        for i in range(0, len(self.input_args)):
            inputs += f'  input_ptrs.emplace_back(static_cast<void *>(input_ptr{str(i)}));\n'
            
        outputs = ''
        for i in range(0, len(self.output_args)):
            outputs += f'  output_ptrs.emplace_back(output_ptr{str(i)});\n'
       
        self.run.splice(
            f"""
                extern "C" void run({input_paras}
                    {output_paras}) {'{'}
                    // stage 3: input and output
                    std::vector<void *> input_ptrs;
                    {inputs}
                    std::vector<void *> output_ptrs;
                    {outputs}
                    // stage 4: run
                    run(exe_ptr, input_ptrs, output_ptrs);
                {'}'}
                \''')\n
                
                async_compile.wait(globals())
                del async_compile
            """
        )
        
        return self.run.getvalue()
    

    def generate_cal(self):
        args = []
        args_str = ''
        del_args = ''
        call_str = '    kernel_cpp_0('
        for i in range(0, len(self.input_args)):
            args.append('arg' + str(i) + '_1')
            args_str += '    arg' + str(i) + '_1 = ' + (f"rand_strided(({str(self.input_args[i].meta['val'].shape).split('[')[-1].split(']')[0]}, ), "
                                                        f"{str(self.input_args[i].meta['val'].stride())}, "
                                                        f"device='{self.input_args[i].meta['val'].device.type}', "
                                                        f"dtype={self.input_args[i].meta['val'].dtype.__str__()})\n")
            
            del_args += '    del arg' + str(i) + '_1\n' 
            call_str += 'c_void_p(arg' + str(i) + '_1.data_ptr()), '
            
        if "(, )" in args_str:
            args_str = args_str.replace("(, )", "()")
        
        bufs = []
        buf_str = ''
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                if i != len(self.output_args) - 1:
                    call_str += ', '
                else:
                    call_str += ')\n'
                bufs.append('buf' + str(i))
                buf_str += '    buf' + str(i) + ' = ' + (f"empty_strided((, ), "
                                                         f"{'(, )'})")
                continue
            call_str += 'c_void_p(buf' + str(i) + '.data_ptr())'
            if i != len(self.output_args) - 1:
                call_str += ', '
            else:
                call_str += ')\n'
                
            bufs.append('buf' + str(i))
            
            buf_str += '    buf' + str(i) + ' = ' + (f"empty_strided(({str(self.output_args[i].meta['val'].shape).split('[')[-1].split(']')[0]}, ), "
                                                     f"{str(self.output_args[i].meta['val'].stride())}, "
                                                     f"device='{self.output_args[i].meta['val'].device.type}', "
                                                     f"dtype={self.output_args[i].meta['val'].dtype.__str__()})\r\n")
        
        if "(, )" in buf_str:
            buf_str = buf_str.replace("(, )", "()")
        
        self.call.splice(
            f"""
                def call(args):
                {', '.join(args)}, = args
                args.clear()
                
                {buf_str}
                {call_str}
                {del_args}
                
                return ({', '.join(bufs)})\n\n'''
                
                if __name__ == "__main__":
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                
                {args_str}
                print(call([{', '.join(args)}]))\n\n
            """
        )
        
        return self.call.getvalue()
        

def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
        return 0
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    raise AssertionError(reduction_type)


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


def reduction_combine_vec(reduction_type, var, next_value):
    if reduction_type == "max":
        return f"{var} = at::vec::maximum({var}, {next_value})"
    elif reduction_type == "min":
        return f"{var} = at::vec::minimum({var}, {next_value})"
    elif reduction_type == "sum":
        return f"{var} += {next_value}"
    else:
        raise NotImplementedError()


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    if reduction_type == "argmax":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmax : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    elif reduction_type == "argmin":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmin : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    return prefix


def float16_reduction_prefix(rtype):
    # TODO: This user-defined reduction uses float16 accumulation for sum. To reduce numerical
    # errors, float32 accumulation should be used instead.
    assert rtype in (
        "sum",
        "any",
    ), f"float16 user-defined reduction only supports 'sum' and 'any' but got {rtype}"
    prefix = [
        f"#pragma omp declare reduction({RTYPE_TO_CPP[rtype]}:{DTYPE_TO_CPP[torch.float16]}:"
        + f"omp_out = omp_out {RTYPE_TO_CPP[rtype]} omp_in)"
    ]
    return prefix


def parallel_num_threads():
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.lru_cache()
def cpp_prefix():
    path = Path(__file__).parent / "cpp_prefix.h"
    with path.open() as f:
        _, filename = codecache.write(
            f.read(),
            "h",
        )
    return f'#include "{filename}"'

class EnflameOverrides(OpOverrides):
    """Map element-wise ops to C++"""
    
    count = 0
    
    @staticmethod
    def para_args(args_dict, node, args):
        src_code = ''
        args_str = []
        for i in range(0, len(args)):
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            elif isinstance(args[i], bool):
                args_str.append(str(args[i]).lower())
            elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
                args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            elif isinstance(args[i], torch.dtype):
                in_shape_size = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                src_code = f'  std::vector<int64_t> type{EnflameOverrides.count}_in_shape{in_shape_size};\n'
                src_code += f'  builder::Type type{EnflameOverrides.count} = builder::Type(type{EnflameOverrides.count}_in_shape, ptype);\n\n'
                args_str.append(f'type{EnflameOverrides.count}')
                EnflameOverrides.count += 1
            else:          
                # if 'squeeze' in operation:
                if True:
                    src_code = f'  builder::Type axes{str(EnflameOverrides.count)}_type({"{" + "1" + "}"}, builder::PrimitiveType::S64());\n'
                    src_code += f'  std::vector<int64_t> axes{_count}_data = {"{" + str(args[i]).split("[")[-1].split("]")[0] + "}"};\n'
                    src_code += f'  builder::Op axes{EnflameOverrides.count} = builder::Const(hlir_builder, (axes{EnflameOverrides.count}_data.data()), axes{EnflameOverrides.count}_type);\n'
                    args_str.append(f'axes{EnflameOverrides.count}')
                    shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                    src_code += f"  builder::Type output{EnflameOverrides.count}_type({shape}, builder::PrimitiveType::F32());\n"
                    args_str.append(f"output{EnflameOverrides.count}_type")
                    EnflameOverrides.count +=1
                else:                                  
                    in_shape_size = '{1}'
                    if isinstance(type(args[i]), type(int)):
                        src_code = f'  int value{EnflameOverrides.count} = {str(args[i])};\n'
                    else:
                        src_code = f'  float value{EnflameOverrides.count} = {str(args[i])};\n'
                    src_code += f'  std::vector<int64_t> const{EnflameOverrides.count}_in_shape{in_shape_size};\n'
                    src_code += f'  builder::Type value{EnflameOverrides.count}_type(const{EnflameOverrides.count}_in_shape, ptype);\n'
                    src_code += f'  builder::Op const{EnflameOverrides.count} = builder::Const(hlir_builder, static_cast<void *>(&value{EnflameOverrides.count}), value{EnflameOverrides.count}_type);\n\n'
                    args_str.append(f'const{EnflameOverrides.count}')
                    EnflameOverrides.count += 1

        return src_code, args_str

    @staticmethod
    def gen(operation, args_dict, node, args):
        src_code, args_str = EnflameOverrides.para_args(args_dict, node, args)
        src_code += f"  builder::Op {args_dic[node.name]} = builder::{operation}({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Add(args_dict, node, args):
        return EnflameOverrides.gen('Add', args_dict, node, args)

    @staticmethod
    def Sub(args_dict, node, args):
        return EnflameOverrides.gen('Sub', args_dic, node, args)

    @staticmethod
    def Sqrt(args_dict, node, args):
        return EnflameOverrides.gen('Sqrt', args_dic, node, args)
        
    @staticmethod
    def Clone(args_dict, node, args):
        return f"  builder::Op {args_dic[node.name]} = {args_dict[args[0].name]};\n"
    
    @staticmethod
    def ReduceMean(args_dict, node, args):
        args_str = EnflameOverrides.para_args(args_dict, node, args)
        args_str[1], args_str[2] = args_str[2], args_str[1]
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f'  builder::Type reducemean_shape{EnflameOverrides.count}({shape}, ptype);\n'
        args_str.append(f'reducemean_shape{EnflameOverrides.count}')
        for i in range(0, len(args_str)):
            print(args_str[i])
        tmp = args[1].copy()
        tmp.sort()
        for i in range(0, len(tmp)):
            tmp[i] = (tmp[i] + len(args[0].meta['val'].shape)) % len(args[0].meta['val'].shape)
        args_str[2] = str(tmp).replace('[', '{').replace(']', '}')
        args_str[2] = '{2, 3}'
        EnflameOverrides.count += 1
        src_code += f"  builder::Op {args_dict[node.name]} = builder::ReduceMean({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Reshape(args_dict, node, args):
        src_code, args_str = EnflameOverrides.para_args(args_dict, node, args)
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code += f'  builder::Type reshape_shape{EnflameOverrides.count}({shape}, ptype);\n'
        args_str[1] = f'reshape_shape{EnflameOverrides.count}'
        EnflameOverrides.count += 1
        
        src_code += f"  builder::Op {args_dict[node.name]} = builder::Reshape({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Addmm(args_dict, node, args):
        src_code = f"  builder::Op addmm{EnflameOverrides.count} = builder::Gemm({'{' + args_str[1] + ', ' + args_str[2] + '}'});\n"
        src_code += f"  builder::Op {args_dict[node.name]} = builder::Add({args_str[0]}, addmm{EnflameOverrides.count});\n"
        EnflameOverrides.count += 1
        return src_code
    
    @staticmethod
    def ReduceMax(args_dict, node, args):
        args_str = EnflameOverrides.para_args(args_dict, node, args)
        if len(args_str) ==3:
            args_str[1], args_str[2] = args_str[2], args_str[1]
        src_code = f"  builder::Op {args_dict[node.name]} = builder::ReduceMax({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def ReduceSum(args_dict, node, args):
        args_str = EnflameOverrides.para_args(args_dict, node, args)
        if len(args_str) ==3:
            args_str[1], args_str[2] = args_str[2], args_str[1]
        src_code = f"  builder::Op {args_dict[node.name]} = builder::ReduceSum({', '.join(args_str)});\n\n"
        return src_code   
    
    @staticmethod
    def Gemm(args_dict, node, args):
        args_str = EnflameOverrides.para_args(args_dict, node, args)
        src_code = f"  builder::Op {args_dict[node.name]} = builder::Gemm({'{' + ', '.join(args_str) + '}'});\n\n"
        return src_code

    @staticmethod
    def Transpose(args_dict, node, args):
        args_str = EnflameOverrides.para_args(args_dict, node, args)
        if len(args) == 1:
            args_str.append('{1, 0}')
        src_code = f"  builder::Op {args_dict[node.name]} = builder::Transpose({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Getitem(args_dict, node, args):
        if 'max_pool2d' in args[0].name:
            args_dict[node.name] = args_dict[args[0].name]
            return ''
        
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f'  builder::Type getitem_type{EnflameOverrides.count}({shape}, ptype);\n\n'
        src_code += f"  builder::Op {args_dict[node.name]} = builder::GetTupleElement({args_dict[args[0].name]}, {int(node.args[1])}, getitem_type{count});\n\n"
        EnflameOverrides.count += 1
        return src_code
    
    @staticmethod
    def Gather(args_dict, node, args):
        args_str = EnflameOverrides.para_args(args_dict, node, args)
        
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f'  builder::Type gather_type{EnflameOverrides.count}({shape}, ptype);\n\n'
        
        src_code += f"  std::vector<int64_t> gather_offset_dim{EnflameOverrides.count};\n"
        src_code += f"  for (int64_t i = 0; i < {args[1]}; i++) {'{'}\n     gather_offset_dim{EnflameOverrides.count}.emplace_back(i);\n  {'}'}\n\n"
        src_code += f"  auto gather_data_shape{EnflameOverrides.count} = {args_str[0]}.GetType().GetShape();\n"
        src_code += f"  auto gather_indices_shape{EnflameOverrides.count} = {args_str[2]}.GetType().GetShape();\n"
        src_code += f"  for (int64_t i = {args[1]} + 1; i < gather_data_shape{EnflameOverrides.count}.size(); i++) {'{'}\n    gather_offset_dim{EnflameOverrides.count}.emplace_back(i - 1 + gather_indices_shape{EnflameOverrides.count}.size());\n  {'}'}\n"
        src_code += f"  std::vector<int64_t> gather_slice_size{EnflameOverrides.count}(gather_data_shape{EnflameOverrides.count});\n"
        src_code += f"  gather_slice_size{EnflameOverrides.count}[{args[1]}] = 1;\n\n"
        
        src_code += f"  builder::GatherDimensionNumbers gather_gnums{EnflameOverrides.count}(gather_offset_dim{EnflameOverrides.count}, {'{' + '1' + '}'}, {'{' + '1' + '}'}, gather_indices_shape{EnflameOverrides.count}.size());\n"
        
        src_code += f"  auto {args_dict[node.name]} = builder::Gather({args_str[0]}, {args_str[2]}, gather_gnums{EnflameOverrides.count}, gather_slice_size{EnflameOverrides.count}, false, gather_type{EnflameOverrides.count});\n"
        
        EnflameOverrides.count += 1

        return src_code

    @staticmethod
    def Batch_Norm(args_dict, node, args):
        shape = []
        lenth = len(node.meta['val'])
        for i in range (0, lenth):
            shape.append('{' + str(node.meta['val'][i].shape).split('[')[-1].split(']')[0] + '}')
        
        src_code = f"  std::vector<std::vector<int64_t>> tuple_shape{EnflameOverrides.count};\n"
        src_code += f"  std::vector<builder::PrimitiveType> tuple_dtype{EnflameOverrides.count};\n"
        
        # for i in range(0, lenth):
        #     src_code += f"    tuple_shape{self.bn_count}.push_back({shape[i]});\n"
        # src_code += f"  for (uint i = 0; i < {lenth}; i++) {'{'}\n"
        # # src_code += f"    tuple_shape{self.bn_count}.push_back({shape[i]});\n"
        # src_code += f"    tuple_dtype{self.bn_count}.push_back(builder::PrimitiveType::F32());\n  {'}'}\n"
        
        src_code += f"  builder::Type bn_type{EnflameOverrides.count}(tuple_shape{EnflameOverrides.count}, tuple_dtype{EnflameOverrides.count});\n"
        
        # src_code += f"  auto {self.args_dict[name]} = builder::BatchNormInference({args_str[0]}, {args_str[1]}, {args_str[2]}, {args_str[3]}, {args_str[4]}, 0.1, 3);\n"
        # src_code += f"  auto {self.args_dict[name]} = builder::BatchNormTraining({args_str[0]}, {args_str[1]}, {args_str[2]}, 0.1, 5);\n"
        
        src_code += f"  auto {args_dict[node.name]} = enflame::batch_norm(hlir_builder, {args_str[0]}, {args_str[1]}, {args_str[2]});\n"     
        EnflameOverrides.count += 1
        
        return src_code
    
    @staticmethod
    def Threshold_Backward(args_dict, node, args):
        src_code = f"  builder::Op {args_dict[node.name]} = builder::ReluGrad({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Conv2D(args_dict, node, args):
        args_str =[]
        for i in range(0, 3):
            if isinstance(args[i], type(None)):
                continue
            args_str.append(self.args_dict[args[i].name])
                
        stride = str(args[3]).replace('[','{').replace(']','}')
        
        if len(args[4]) == 1:
            row_padding, col_padding = args[4][0]
        else:
            row_padding = args[4][0]
            col_padding = args[4][1]
        
        padding = f"{'{' + str(row_padding)}, {str(row_padding)}, {str(col_padding)}, {str(col_padding) + '}'}"
        
        dilation = str(args[5]).replace('[','{').replace(']','}')
        group = args[8]
    
        ssrc_code = f"  std::vector<builder::Op> {args_dict[node.name]}_inputs = {'{' + ', '.join(args_str) + '}'};\n"
        src_code += f'  builder::Op {args_dict[node.name]} = builder::Conv2D({self.args_dict[name]}_inputs, {group}, "NOTSET", "NCHW", {stride}, {padding}, {dilation});\n\n'
        
        return src_code
    
    @staticmethod
    def Conv2D_Grad(args_dict, node, args):
        args_str = []

        for i in range(0, 3):
            if isinstance(args[i], type(None)):
                continue
            else:
                args_str.append(args_dict[args[i].name])
                
        bias = str(args[3]).replace('[','{').replace(']','}')
        stride = str(args[4]).replace('[','{').replace(']','}')
        padding = str(args[5]).replace('[','{').replace(']','}')
        dilation = str(args[6]).replace('[','{').replace(']','}')
        
        src_code = f"  auto {args_dict[node.name]} = enflame::conv2d_grad(hlir_builder, {args_str[0]}, {args_str[1]}, {args_str[2]}, {bias}, {stride}, {padding}, {dilation});\n"
        
        return src_code
    
    @staticmethod            
    def Max_Pool2D(args_dict, node, args):
        ceil_mode = 'false'
        return_indices = 'false'
        padding = '{0, 0, 0, 0}'
        dilation = '{1, 1}'
        shape = '{' + str(node.meta['val'][0].shape).split('[')[-1].split(']')[0] + '}'
        dtype = node.meta['val'][0].dtype
        
        src_code = f'  builder::Type max_pool_type{EnflameOverrides.count} = builder::Type({shape}, ptype);\n\n'
        
        if len(args) == 3:
            ksize = str(args[1]).replace('[','{').replace(']','}')
            stride = str(args[2]).replace('[','{').replace(']','}')
        else:
            ksize = str(args[1]).replace('[','{').replace(']','}')
            stride = str(args[2]).replace('[','{').replace(']','}')            

        src_code += f'  builder::Op {args_dict[node.name]} = builder::MaxPool2D({args_dict[args[0].name]}, {ksize}, {ceil_mode}, {return_indices}, "NOTSET", "NCHW", {stride}, {padding}, {"{" + "}"}, max_pool_type{EnflameOverrides.count});\n'
        
        EnflameOverrides.count += 1
        return src_code
    
    @staticmethod
    def Max_Pool2D_Grad(args_dict, node, args):
        ksize = str(args[2]).replace('[','{').replace(']','}')
        strides = str(args[3]).replace('[','{').replace(']','}')
        padding = str(args[4]).replace('[','{').replace(']','}')
        
        src_code += f"  auto {args_dict[node.name]} = enflame::max_pool2d_grad(hlir_builder, {args_dict[args[0].name]}, {args_dict[args[1].name]}, {ksize}, {strides}, {padding});\n"
        
        return src_code
