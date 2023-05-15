import contextlib
import dataclasses
import functools
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List
from io import StringIO
import textwrap
from torch._inductor.utils import IndentedBuffer

import torch

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate

from torch._inductor.codegen.common import OpOverrides


type_set = {"torch.float32": "builder::PrimitiveType::F32()",
            "torch.int64": "builder::PrimitiveType::S64()",
            "torch.bool": "builder::PrimitiveType::PRED()"}

need_node = ['Scalar', 'Reshape', 'Expand', 'Zeros', 'Fulllike', 'Getitem', 'Gather', 'Scatter', 'Batch_Norm', 
             'Convolution', 'Conv2D_Grad', 'MaxPool2D', 'MaxPool2D_Grad']

def process_name(name, target):
    if target.__name__ == 'convolution_backward':
            return 'Conv2D_Grad'
    if target.__name__ == 'max_pool2d_with_indices':
        return 'MaxPool2D'
    if target.__name__ == 'max_pool2d_with_indices_backward':
        return 'MaxPool2D_Grad'
    
    if hasattr(target, "name"):
        real_op = target.name().split('::')[-1]
        if real_op.find('.') != -1:
            real_op = real_op.split('.')[0]
    else:
        real_op = name.rsplit('_', 1)[0] if name[-1].isdigit() else name

    return real_op.title()

class EnflameCodegen(torch.fx.Interpreter):
    def __init__(self, graph):
        self.name = 'topsgraph'
        self.import_code = IndentedBuffer()
        
        self.args_dict = {}
        self.input_args =[]
        self.output_args = []
        self.build_graph_code = IndentedBuffer(initial_indent=1)
        
        self.graph = graph
        super().__init__(graph)
        self.override = EnflameOverrides

    def placeholder(self, name, target, args, kwargs):    
        self.args_dict[name] = 'op' + str(len(self.args_dict))
        self.input_args.append(self.cur_node)
        
        data_type = self.cur_node.meta['val'].dtype.__str__()
        if data_type not in type_set.keys():
            print(data_type, flush=True)
            raise ValueError("Type error")
    
        in_shape = self.get_shape()
        if in_shape == '{}':
            in_shape = '{1}'
        self.build_graph_code.writeline(f'std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape};')
        self.build_graph_code.writeline(f'builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, {type_set[data_type]});')
        self.build_graph_code.writeline(f'builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);\n')

    def call_function(self, name, target, args, kwargs):   
        if name not in self.args_dict.keys():
            self.args_dict[name] = 'op' + str(len(self.args_dict))

        arg_code, args_list = EnflameOverrides.gen_args(self.args_dict[name], self.args_dict, self.cur_node, args)
        real_op = process_name(name, target)
        
        # print("*******************Debug info*******************", flush=True)
        print("name:", name, flush=True)
        print("target:", target.name(), flush=True)
        print("real_op:", real_op, flush=True)
        print("args:", args, flush=True)
        print("arg_code:", arg_code.getvalue(), flush=True)
        print("args_list:", args_list, flush=True)

        op_code = getattr(self.override, real_op)(*args_list)
        
        print("op_code:", op_code, flush=True)
        
        self.build_graph_code.splice(arg_code)
        self.build_graph_code.splice(f'{op_code}')
        return
    
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
        test = self.generate_code()
        with open('codegen.py', 'w') as f:
            f.write(test)
        
        print("*******************Generated code*******************")
        print(test, flush=True)

        return test

    def get_shape(self):
        shape = '{' + str(self.cur_node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        return shape

    def gen_import_code(self):
        self.import_code.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import random
                from torch import empty_strided, as_strided, device
                from dicp.TopsGraph.compile import AsyncCompileTopsGraph
            """
            , strip=True
        )
        return self.import_code.getvalue()

    def gen_build_graph_code(self):
        graph_code = IndentedBuffer()
        graph_code.writelines(
            [
                f'auto hlir_builder = std::make_shared<builder::Builder>();',
                f'hlir_builder->SetShapeInference(true);',
                f'auto ptype = builder::PrimitiveType::F32();\n'
            ]
        )

        graph_code.splice(self.build_graph_code, strip=True)
        
        output_str = []
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                continue
            else:
                output_str.append(self.args_dict[self.output_args[i].name])

        graph_code.writeline(f'hlir_builder->SetOutput({"{" + ", ".join(output_str) + "}"});\n')
        graph_code.writeline(f'return hlir_builder;')
        return graph_code
    
    def gen_compile_func_code(self):
        compile_func_body = IndentedBuffer()
        with compile_func_body.indent():
            compile_func_body.splice(
                f"""
                    auto hlir_builder = build_sample();
                    compile(hlir_builder, &exe_ptr);
                """
                , strip=True
            )
        compile_func = IndentedBuffer()
        compile_func.writelines(
            [
                f'topsExecutable_t exe_ptr;',
                f'extern "C" void compile(void){"{"}'
            ]
        )
        with compile_func.indent():
            compile_func.splice(compile_func_body)
        compile_func.writeline('}')

        return compile_func
                
    def gen_run_func_code(self):
        func_body = IndentedBuffer()
        func_body.writeline(f'std::vector<void *> input_ptrs;')
        for i in range(0, len(self.input_args)):
            func_body.writeline(f'input_ptrs.emplace_back(static_cast<void *>(input_ptr{str(i)}));')
        func_body.writeline(f'std::vector<void *> output_ptrs;')
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                func_body.writeline(f'output_ptrs.emplace_back(output_ptr{str(i)});')
        func_body.writeline(f'run(exe_ptr, input_ptrs, output_ptrs);')

        input_paras = ''
        for i in range(0, len(self.input_args)):
            input_paras += f'float* input_ptr{str(i)}, '
        output_paras = []
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                output_paras.append(f'float* output_ptr{str(i)}')
        output_paras = ', '.join(output_paras)

        run_func_code = IndentedBuffer()
        run_func_code.writeline(f'extern "C" void run({input_paras} {output_paras}) {"{"}')
        with run_func_code.indent():
            run_func_code.splice(func_body)
        run_func_code.splice('}')
        return run_func_code

    def get_kernel_header(self):
            return f"""
                    #include <cmath>
                    #include <fstream>
                    #include <iostream>
                    #include <sstream>
                    #include <string>
                    #include <vector>

                    #include "dtu_utils.h"
                    #include "common_ops.h"
                    #include "conv2d_grad.h"
                    #include "maxpool2d_grad.h"

                    #include "dtu/hlir_builder/hlir_builder.h"
                    #include "dtu/hlir_builder/hlir_builder_client_ops.h"
                """

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        compile_graph_code.splice(
            f"""
                async_compile = AsyncCompileTopsGraph()
                kernel_cpp_0 = async_compile.topsgraph('''
            """
            , strip=True
        )
        compile_graph_code.splice(self.get_kernel_header(), strip=True)
        compile_graph_code.writeline(f'std::shared_ptr<builder::Builder> build_sample() {"{"}')
        with compile_graph_code.indent():
            compile_graph_code.splice(self.gen_build_graph_code())
        compile_graph_code.writeline('}')

        compile_graph_code.splice(self.gen_compile_func_code())
        compile_graph_code.splice(self.gen_run_func_code())
        compile_graph_code.writeline(f"''')")
        compile_graph_code.writeline('async_compile.wait(globals())')
        compile_graph_code.writeline('del async_compile')

        return compile_graph_code.getvalue()

    def gen_tensor(self, prefix, tensor):
        res =  f"{prefix}({tuple(tensor.shape)}, {tensor.stride()}, device='{tensor.device.type}', dtype={tensor.dtype})"
        return res

    def gen_empty_tensor(self, tensor):
        return self.gen_tensor("empty_strided", tensor)

    def gen_random_tensor(self, tensor):
        return self.gen_tensor("rand_strided", tensor)

    def gen_call_func(self):
        call_body = IndentedBuffer()

        args = []
        for i in range(len(self.input_args)):
            args.append('arg' + str(i))
        call_body.writeline(f"{', '.join(args)}, = args")
        call_body.writeline(f"args.clear()")

        bufs = []
        for i in range(len(self.output_args)):
            bufs.append('buf' + str(i))
            if isinstance(self.output_args[i], type(None)):
                call_body.writeline(bufs[-1] + ' = ' + (f"empty_strided((), ())"))
            else:
                otensor = self.output_args[i].meta['val']
                call_body.writeline(bufs[-1] + ' = ' + self.gen_empty_tensor(otensor))

        call_str = 'kernel_cpp_0('
        for i in range(len(self.input_args)):
            call_str += 'c_void_p(' + args[i] + '.data_ptr()), '
        for i in range(len(self.output_args)):
            call_str += 'c_void_p(' + bufs[i] + '.data_ptr())'
            if i != len(self.output_args) - 1:
                call_str += ', '
            else:
                call_str += ')'
        call_body.writeline(call_str)

        for arg in args:
            call_body.writeline(f'del {arg}')
        
        call_body.writeline(f"return ({', '.join(bufs)})")

        call_func =IndentedBuffer()
        call_func.writeline(f"def call(args):")
        with call_func.indent():
            call_func.splice(call_body)
        return call_func.getvalue()

    def gen_main_func(self):
        main_body = IndentedBuffer()
        main_body.splice(
            f"""
            from torch._dynamo.testing import rand_strided
            from torch._inductor.utils import print_performance
            """
            , strip=True
        )
        for i in range(0, len(self.input_args)):
            itensor = self.input_args[i].meta['val']
            main_body.writeline('arg' + str(i) + ' = ' + self.gen_random_tensor(itensor))

        args = []
        for i in range(len(self.input_args)):
            args.append('arg' + str(i))
        main_body.writeline(f"print(call([{', '.join(args)}]))")

        main_func = IndentedBuffer()
        main_func.writeline(f"""if __name__ == "__main__":""")
        with main_func.indent():
            main_func.splice(main_body)
        return main_func.getvalue()

    def generate_code(self):
        return (self.gen_import_code() + self.gen_compile_graph_code()+ self.gen_call_func() + self.gen_main_func())

class EnflameOverrides(OpOverrides):
    @staticmethod
    def gen_args(op_var, args_dict, node, args):
        gen_const_flag = True
        src_code = IndentedBuffer()
        args_str = [op_var]
        count = 0
        if process_name(node.name, node.target) in need_node:
            print('process_name(node.name, node.target):', process_name(node.name, node.target), flush=True)
            gen_const_flag = False
            args_str.append(node)

        for i in range(len(args)):
            if isinstance(args[i], type(None)):
                continue
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            elif isinstance(args[i], bool):
                args_str.append(str(args[i]).lower())
            elif isinstance(args[i], torch.dtype):
                shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                data_type = node.meta['val'].dtype.__str__()
                src_code.writeline(f'std::vector<int64_t> {op_var}_shape{count}{shape};')
                src_code.writeline(f'builder::Type {op_var}_type{count} = builder::Type({op_var}_shape{count}, {type_set[data_type]});')
                args_str.append(f'{op_var}_type{count}')
                count += 1
            elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
                if "reducemean" in node.name and len(args) != 1 and i == 1:
                    tmp = args[1].copy()
                    tmp.sort()
                    for j in range(0, len(tmp)):
                        tmp[j] = (tmp[j] + len(args[0].meta['val'].shape)) % len(args[0].meta['val'].shape)
                    args_str.append(str(tmp).replace('[', '{').replace(']', '}'))
                else:
                    args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            else:
                if "squeeze" in node.name:
                    src_code.writeline(f'builder::Type {op_var}_axes_type{count}({"{" + "1" + "}"}, builder::PrimitiveType::S64());')
                    
                    if "unsqueeze" in node.name:
                        src_code.writeline(f'std::vector<int64_t> {op_var}_axes_data{count} = {"{" + str(args[i]).split("[")[-1].split("]")[0] + "}"};')
                    
                    else:
                        src_code.writeline(f'std::vector<int64_t> {op_var}_axes_data{count} = {"{" + str(args[i]) + "}"};\n')

                    src_code.writeline(f'builder::Op {op_var}_axes{count} = builder::Const(hlir_builder, ({op_var}_axes_data{count}.data()), {op_var}_axes_type{count});')
                    args_str.append(f'{op_var}_axes{count}')
                    shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                    data_type = node.meta['val'].dtype.__str__()
                    src_code.writeline(f"builder::Type {op_var}_output_type{count}({shape}, {type_set[data_type]});\n")
                    args_str.append(f"{op_var}_output_type{count}")
                    count += 1
                elif gen_const_flag:
                    if isinstance(node.meta['val'], list) or isinstance(node.meta['val'], tuple):
                        val = node.meta['val'][0]
                    else:
                        val = node.meta['val']

                    data_type = '' if isinstance(val, type(None)) else val.dtype.__str__()
                
                    if data_type == 'torch.int64':
                        out_type = 'builder::PrimitiveType::S64()'
                        src_code.writeline(f'int {op_var}_const_value{count} = {str(args[i])};')
                    else:
                        out_type = 'builder::PrimitiveType::F32()'
                        src_code.writeline(f'float {op_var}_const_value{count} = {str(args[i])};')
  
                    src_code.writeline(f"builder::Type {op_var}_const_value_type{count}({'{' + '1' + '}'}, {out_type});")
                    src_code.writeline(f'builder::Op {op_var}_const{count} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value{count}), {op_var}_const_value_type{count});\n')
                    args_str.append(f'{op_var}_const{count}')
                    count += 1

        return src_code, args_str

    @staticmethod
    def Clone(op_var, x):
        return f"builder::Op {op_var} = {x};"

    @staticmethod
    def Abs(op_var, x):
        return f"builder::Op {op_var} = builder::Abs({x});"

    @staticmethod
    def Add(op_var, x, y):
        return f"builder::Op {op_var} = builder::Add({x}, {y});"
 
    @staticmethod
    def Sub(op_var, x, y):
        return f"builder::Op {op_var} = builder::Sub({x}, {y});"
    
    @staticmethod
    def Mul(op_var, x, y):
        return f"builder::Op {op_var} = builder::Mul({x}, {y});"
    
    @staticmethod
    def Div(op_var, x, y):
        return f"builder::Op {op_var} = builder::Div({x}, {y});"
    
    @staticmethod
    def Gemm(op_var, x, y):
        return f"builder::Op {op_var} = builder::Gemm({'{' + x + ',' + y + '}'});"
    
    @staticmethod
    def Lessequal(op_var, x, y):
        return f'builder::Op {op_var} = builder::LessEqual({x}, {y});'
    
    @staticmethod
    def Log(op_var, x):
        return f"builder::Op {op_var} = builder::Log({x});"
    
    @staticmethod
    def Neg(op_var, x):
        return f"builder::Op {op_var} = builder::Neg({x});"
    
    @staticmethod
    def Pow(op_var, x, y):
        return f"builder::Op {op_var} = builder::Pow({x}, {y});"
    
    @staticmethod
    def Exp(op_var, x):
        return f"builder::Op {op_var} = builder::Exp({x});"

    @staticmethod
    def Sqrt(op_var, x):
        return f"builder::Op {op_var} = builder::Sqrt({x});"
     
    @staticmethod
    def Relu(op_var, x):
        return f"builder::Op {op_var} = builder::Relu({x});"
    
    @staticmethod
    def Convert(op_var, *args):
        return f"builder::Op {op_var} = builder::Convert({', '.join(args)});"
    
    @staticmethod
    def Reciprocal(op_var, *args):
        return f"builder::Op {op_var} = builder::Reciprocal({', '.join(args)});"
    
    @staticmethod
    def Rsqrt(op_var, x):
        return f"builder::Op {op_var} = builder::Rsqrt({x});"
    
    @staticmethod
    def Scalar(op_var, node, *args_str):
        src_code = f"builder::Type {op_var}_scalar_type({'{' + '1' +'}'}, builder::PrimitiveType::F32());\n"
        src_code += f"std::vector<float> {op_var}_scalar_data(1, {node.args[0]});\n"
        src_code += f"auto {op_var} = builder::Const(hlir_builder, static_cast<void *>({op_var}_scalar_data.data()), {op_var}_scalar_type);\n"
        return src_code
    
    @staticmethod
    def Getitem(op_var, node, *args_str):
        src_code = f"builder::Op {op_var} = builder::GetTupleElement({args_str[0]}, {int(node.args[1])});\n"
        return src_code
    
    @staticmethod
    def Select(op_var, *args):
        return f"builder::Op {op_var} = builder::Select({', '.join(args)});"

    @staticmethod
    def Zeros(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()            
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"builder::Op {op_var} = builder::ZerosLike({args_str[0]}, {type_set[data_type]}, {shape});\n\n"
        return src_code
    
    @staticmethod
    def Fulllike_tmp(op_var, node, *args_str):
        src_code = f"builder::Type {op_var}_type({'{' + '1' + '}'}, builder::PrimitiveType::F32());\n"
        src_code += f"std::vector<float> {op_var}_data = {'{' + str(node.args[1]) + '}'};\n"
        src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, ({op_var}_data.data()), {op_var}_type);\n"
        return src_code
    
    @staticmethod
    def Fulllike(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"  builder::Op {op_var} = builder::ZerosLike({args_str[0]}, {type_set[data_type]}, {shape});\n\n"
        return src_code
    
    @staticmethod
    def Transpose(op_var, *args):
        if len(args) == 1:
            list(args).append('{1, 0}')
        return f"builder::Op {op_var} = builder::Transpose({', '.join(args)});\n"
    
    @staticmethod
    def Reshape(op_var, node, *args_str):
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f'builder::Type {op_var}_reshape_shape({shape}, ptype);\n'
        tmp = f'{op_var}_reshape_shape'
        src_code += f"builder::Op {op_var} = builder::Reshape({args_str[0]}, {tmp});\n\n"
        return src_code
    
    @staticmethod
    def Expand(op_var, node, *args_str):
        dims = []
        for i in range(0, len(node.meta['val'].shape)):
            dims.append(str(i))
        broadcast_dims = '{' + ', '.join(dims) + '}'
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        src_code = f'builder::Type {op_var}_expand_type({shape}, {type_set[data_type]});\n'
        src_code += f"auto {op_var} = BroadcastInDim({args_str[0]}, {broadcast_dims}, {op_var}_expand_type);\n"
        return src_code
    
    @staticmethod
    def Squeeze(op_var, *args):
        return f"builder::Op {op_var} = builder::Squeeze({', '.join(args)});"
    
    @staticmethod
    def Unsqueeze(op_var, *args):
        return f"builder::Op {op_var} = builder::Unsqueeze({', '.join(args)});"

    @staticmethod
    def Reducemean(op_var, *args):
        src_code = ''
        if len(args) == 3:
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args[0]}, {args[2]}, {args[1]});\n"
        elif len(args) == 2:
            keepdim = 'false'
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args[0]}, {keepdim}, {args[1]});\n"
        elif len(args) == 1:
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args[0]});\n"
        else:   
            ValueError("Reducemean args num error!")
        return src_code

    @staticmethod
    def Reducemax(op_var, *args):
        src_code = ''
        if len(args) == 3:
            src_code = f"builder::Op {op_var} = builder::ReduceMax({args[0]}, {args[2]}, {args[1]});\n"
        elif len(args) == 2:
            keepdim = 'false'
            src_code = f"builder::Op {op_var} = builder::ReduceMax({args[0]}, {keepdim}, {args[1]});\n"
        elif len(args) == 1:
            src_code = f"builder::Op {op_var} = builder::ReduceMax({args[0]});\n"
        else:
            ValueError("ReduceMax args num error!")
        return src_code

    @staticmethod
    def Reducesum(op_var, *args):
        src_code = ''
        if len(args) == 3:
            src_code = f"builder::Op {op_var} = builder::ReduceSum({args[0]}, {args[2]}, {args[1]});\n"
        elif len(args) == 2:
            keepdim = 'false'
            src_code = f"builder::Op {op_var} = builder::ReduceSum({args[0]}, {keepdim}, {args[1]});\n"
        elif len(args) == 1:
            src_code = f"builder::Op {op_var} = builder::ReduceSum({args[0]});\n"
        else:
            ValueError("ReduceSum args num error!")
        return src_code

    @staticmethod
    def Scatter(op_var, node, *args_str):
        new_args_str = []
        new_args_str.append(args_str[0])
        new_args_str.append(str(node.args[1]))
        new_args_str.append(args_str[1])
        
        if isinstance(node.args[3], float):
            src_code = f'const float {op_var}_value = {str(node.args[3])};\n'
        else:
            src_code = f'const int {op_var}_value = {str(node.args[3])};\n'
        
        new_args_str.append(f'{op_var}_value')

        src_code += f"auto {op_var} = enflame::Scatter(hlir_builder, {', '.join(new_args_str)});\n"

        return src_code
    
    @staticmethod
    def Gather(op_var, node, *args_str):
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        new_args_str = []
        new_args_str.append(args_str[0])
        new_args_str.append(args_str[1])
        new_args_str.append(str(node.args[1]))
        
        src_code = f"builder::Type {op_var}_gather_type({shape}, builder::PrimitiveType::F32());\n"
        new_args_str.append(f"{op_var}_gather_type")
        
        src_code += f"auto {op_var} = enflame::Gather(hlir_builder, {', '.join(new_args_str)});\n"

        return src_code
    
    @staticmethod
    def Batch_Norm(op_var, node, *args_str):
        args_str_tmp = args_str[:5]
        src_code = f"auto {op_var} = enflame::BatchNorm(hlir_builder, {', '.join(args_str_tmp)}, 1, true, {str(node.args[6])}, {str(node.args[7])});\n"     
        return src_code

    @staticmethod
    def Convolution(op_var, node, *args_str):
        tmp_str =[]
        index = 3
        for i in range(0, 3):
            if isinstance(node.args[i], type(None)):
                index -= 1
                continue
            tmp_str.append(args_str[i])
        
        stride = args_str[index]
        
        if len(node.args[4]) == 1:
            row_padding, col_padding = node.args[4][0]
        else:
            row_padding = node.args[4][0]
            col_padding = node.args[4][1]

        padding = f"{'{' + str(row_padding)}, {str(row_padding)}, {str(col_padding)}, {str(col_padding) + '}'}"
        dilation = args_str[index + 2]
        
        group = '1'
        src_code = f"std::vector<builder::Op> {op_var}_inputs = {'{' + ', '.join(tmp_str) + '}'};\n"
        src_code += f'builder::Op {op_var} = builder::Conv2D({op_var}_inputs, {group}, "NOTSET", "NCHW", {stride}, {padding}, {dilation});\n\n'

        return src_code

    @staticmethod
    def Conv2D_Grad(op_var, node, *args_str):
        new_args_str = []
        index = 3
        for i in range(0, 3):
            if isinstance(node.args[i], type(None)):
                index -= 1
                continue
            else:
                new_args_str.append(args_str[i])

        bias = args_str[index]
        stride = args_str[index + 1]
        padding = args_str[index + 2]
        dilation = args_str[index + 3]
        
        src_code = f"auto {op_var} = enflame::Conv2D_Grad(hlir_builder, {','.join(new_args_str)}, {bias}, {stride}, {padding}, {dilation});\n"

        return src_code

    @staticmethod            
    def MaxPool2D(op_var, node, *args_str):
        args = node.args
        
        ceil_mode = 'false'
        return_indices = 'false'
        ksize = str(args[1]).replace('[','{').replace(']','}')
        stride = '{1, 1}'
        padding = '{0, 0, 0, 0}'
        
        shape = '{' + str(node.meta['val'][0].shape).split('[')[-1].split(']')[0] + '}'
        
        if len(args) >= 3:
            stride = str(args[2]).replace('[','{').replace(']','}')
        
        if len(args) >= 4:  
            if len(args[3]) == 1:
                row_padding, col_padding = args[3][0]
            else:
                row_padding = args[3][0]
                col_padding = args[3][1]
            padding = f"{'{' + str(row_padding)}, {str(row_padding)}, {str(col_padding)}, {str(col_padding) + '}'}"
                       
        src_code = f'builder::Op {op_var} = enflame::MaxPool2D(hlir_builder, {args_str[0]}, {ksize}, {stride}, {padding}, {shape});\n'

        return src_code
    
    @staticmethod
    def MaxPool2D_Grad(op_var, node, *args_str):
        args = node.args    
        
        ksize = str(args[2]).replace('[','{').replace(']','}')
        strides = str(args[3]).replace('[','{').replace(']','}')
        padding = str(args[4]).replace('[','{').replace(']','}')
        
        src_code = f"auto {op_var} = enflame::MaxPool2D_Grad(hlir_builder, {args_str[0]}, {args_str[1]}, {ksize}, {strides}, {padding});\n"
        
        return src_code
    