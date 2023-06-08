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
from ..config import tops_debug


type_set = {"torch.float16": "builder::PrimitiveType::F16()",
            "torch.float32": "builder::PrimitiveType::F32()",
            "torch.int64": "builder::PrimitiveType::S64()",
            "torch.bool": "builder::PrimitiveType::PRED()",
            "torch.complex64": "builder::PrimitiveType::F32()"}


need_node = ['Scalar', 'Reshape', 'Expand', 'Zeros', 'Full', 'Fulllike', 'Getitem', 'Gather', 'Scatter',
             'Batch_Norm', 'Convolution', 'Conv2D_Grad', 'MaxPool2D', 'MaxPool2D_Grad',
             'Viewasreal', 'Complexmul', 'Concatenate', 'Softmax', 'Logsoftmax', 'Gelu']

need_args = ['Dot', 'Slice', 'Select', 'Complex']

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
            print("data_type:", data_type, flush=True)
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
        
        if tops_debug:
            print("*******************Debug info*******************")
            print("name:", name)
            print("target:", target.name())
            print("real_op:", real_op)
            print("args:", args)
            print("arg_code:", arg_code.getvalue())
            print("args_list:", args_list)
            print("op_code:", getattr(self.override, real_op)(*args_list))

        op_code = getattr(self.override, real_op)(*args_list)
        
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
        if tops_debug:
            with open('codegen.py', 'w') as f:
                f.write(test)
            print("*******************Generated code*******************")
            print(test)

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
        
        name = process_name(node.name, node.target)
        if name == "Reshape" and "complex" in args[0].name:
            args_str.append(node)
            src_code.writeline(f"builder::Op {args_dict[args[0].name]}_0 = builder::GetTupleElement({args_dict[args[0].name]}, 0);\n")
            src_code.writeline(f"builder::Op {args_dict[args[0].name]}_1 = builder::GetTupleElement({args_dict[args[0].name]}, 1);\n")
            args_str.append(f"{args_dict[args[0].name]}_0")
            args_str.append(f"{args_dict[args[0].name]}_1")
            args_str.append(str(args[1]).replace('[', '{').replace(']', '}'))
            return src_code, args_str
        elif name in need_args:
            args_str.append(node)
            args_str.append(args_dict)
            args_str.append(args)
            return src_code, args_str
        elif name in need_node:
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
                if "reduce" in node.name and len(args) != 1 and i == 1:
                    tmp = args[1].copy()
                    tmp.sort()
                    for j in range(0, len(tmp)):
                        tmp[j] = (tmp[j] + len(args[0].meta['val'].shape)) % len(args[0].meta['val'].shape)
                    args_str.append(str(tmp).replace('[', '{').replace(']', '}'))
                elif any(args[i]) and isinstance(args[i][0], Node):
                     nodelistarg = '{'
                     for i in args[i]:
                        assert(isinstance(i, Node))
                        nodelistarg += ' ' + str(args_dict[i.name]) + ','
                     nodelistarg += '}'
                     nodelistarg = nodelistarg.replace(",}", "}")
                     args_str.append(nodelistarg)
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
                        src_code.writeline(f'int {op_var}_const_value{count} = {str(args[i])};')
                    else:
                        src_code.writeline(f'float {op_var}_const_value{count} = {str(args[i])};')
  
                    src_code.writeline(f"builder::Type {op_var}_const_value_type{count}({'{' + '1' + '}'}, {type_set[data_type]});")
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
    def Dot(op_var, node, args_dict, args):
        args_str = []
        src_code = '\n'

        for i in range(0, len(args)):
            tmp_data_type = args[i].meta['val'].dtype.__str__()
            if tmp_data_type != 'torch.float32':
                tmp_shape = '{' + str(args[i].meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                src_code += f"builder::Type {args_dict[args[i].name]}_dot_type({tmp_shape}, builder::PrimitiveType::F32());\n"
                src_code += f"builder::Op {args_dict[args[i].name]}_tmp = builder::Convert({args_dict[args[i].name]}, {args_dict[args[i].name]}_dot_type);\n"
                args_str.append(f"{args_dict[args[i].name]}_tmp")

        src_code += f"builder::DotDimensionNumbers {op_var}_dims_attr({'{0}'}, {'{0}'}, {'{2}'}, {'{1}'});\n"
        src_code += f"builder::Op {op_var}_tmp = builder::DotGeneral({', '.join(args_str)}, {op_var}_dims_attr);\n"

        data_type = node.meta['val'].dtype.__str__()            
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code += f"builder::Type {op_var}_type({shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var} = builder::Convert({op_var}_tmp, {op_var}_type);\n\n"

        return src_code
    
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
    def Sigmoid(op_var, x):
        return f"builder::Op {op_var} = builder::Sigmoid({x});\n"
    
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
        data_type = node.meta['val'].dtype.__str__()
        src_code = f"builder::Type {op_var}_scalar_type({'{' + '1' +'}'}, {type_set[data_type]});\n"
        src_code += f"std::vector<float> {op_var}_scalar_data(1, {node.args[0]});\n"
        src_code += f"auto {op_var} = builder::Const(hlir_builder, static_cast<void *>({op_var}_scalar_data.data()), {op_var}_scalar_type);\n"
        return src_code
    
    @staticmethod
    def Getitem(op_var, node, *args_str):
        src_code = f"builder::Op {op_var} = builder::GetTupleElement({args_str[0]}, {int(node.args[1])});\n"
        return src_code
    
    @staticmethod
    def Where(op_var, *args):
        return f"builder::Op {op_var} = builder::Select({', '.join(args)});"

    @staticmethod
    def Zeros(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()            
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"builder::Op {op_var} = builder::ZerosLike({args_str[0]}, {type_set[data_type]}, {shape});\n\n"
        return src_code
    
    @staticmethod
    def Full(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        src_code = f"builder::Type {op_var}_type({'{' + '1' + '}'}, {type_set[data_type]});\n"
        src_code += f"std::vector<float> {op_var}_data = {'{' + str(node.args[1]) + '}'};\n"
        src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, ({op_var}_data.data()), {op_var}_type);\n"
        return src_code
    
    @staticmethod
    def Fulllike(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"  builder::Op {op_var} = builder::FullLike({args_str[0]}, {str(node.args[1])}, {type_set[data_type]}, {shape});\n\n"
        return src_code
    
    @staticmethod
    def Transpose(op_var, *args):
        if len(args) == 1:
            list(args).append('{1, 0}')
        return f"builder::Op {op_var} = builder::Transpose({', '.join(args)});\n"
    
    @staticmethod
    def Hardswish(op_var, x):
        return f"builder::Op {op_var} = builder::HardSwish({x});"

    @staticmethod
    def Reshape(op_var, node, *args_str):
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        src_code = f'builder::Type {op_var}_reshape_shape({shape}, {type_set[data_type]});\n'
        tmp = f'{op_var}_reshape_shape'
        if len(args_str) == 2:
            src_code += f"builder::Op {op_var} = builder::Reshape({args_str[0]}, {tmp});\n\n"
        else:
            src_code += f"builder::Op {op_var}_0 = builder::Reshape({args_str[0]}, {tmp});\n\n"
            src_code += f"builder::Op {op_var}_1 = builder::Reshape({args_str[1]}, {tmp});\n\n"
            src_code += f"std::vector<builder::Op> {op_var}_outputs = {'{' + op_var +'_0, ' + op_var + '_1' + '}'};\n"
            src_code += f"builder::Op {op_var} = builder::Tuple({op_var}_outputs);\n\n"
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
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        new_args_str = []
        new_args_str.append(args_str[0])
        new_args_str.append(args_str[1])
        new_args_str.append(str(node.args[1]))
        
        src_code = f"builder::Type {op_var}_gather_type({shape}, {type_set[data_type]});\n"
        new_args_str.append(f"{op_var}_gather_type")
        
        src_code += f"auto {op_var} = enflame::Gather(hlir_builder, {', '.join(new_args_str)});\n"

        return src_code
    
    @staticmethod
    def Slice(op_var, node, args_dict, args):
        shape0 = '{' + str(args[0].meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        if shape != shape0:
            raise ValueError("Slice args error!")

        rank = len(args[0].meta['val'].shape)

        start_indices = [0 for x in range(0, rank)]    
        start_indices = '{' + ', '.join(map(str, start_indices)) + '}'   
        limit_indices = shape
        stride = [1 for x in range(0, rank)]
        stride = '{' + ', '.join(map(str, stride)) + '}'   

        src_code = f"auto {op_var} = builder::Slice({args_dict[args[0].name]}, {start_indices}, {limit_indices}, {stride});\n"

        return src_code
    
    @staticmethod
    def Select(op_var, node, args_dict, args):
        shape = args[0].meta['val'].shape
        rank = len(shape)
        dim = int(args[1])

        start_indices = [0 for i in range(0, rank)]    
        start_indices = '{' + ', '.join(map(str, start_indices)) + '}'   

        limit_indices = [x for x in shape]
        limit_indices[dim] = 0
        limit_indices = '{' + ', '.join(map(str, limit_indices)) + '}'

        stride = [1 for x in range(0, rank)]
        stride = '{' + ', '.join(map(str, stride)) + '}' 

        src_code = f"auto {op_var}_t = builder::Slice({args_dict[args[0].name]}, {start_indices}, {limit_indices}, {stride});\n"

        src_code += f"builder::Type {op_var}_axes_type({'{' + '1' + '}'}, builder::PrimitiveType::S64());\n"
        src_code += f"std::vector<int64_t> {op_var}_axes_data = {'{' + str(dim) + '}'};\n"
        src_code += f"builder::Op {op_var}_axes = builder::Const(hlir_builder, ({op_var}_axes_data.data()), {op_var}_axes_type);\n"
        src_code += f"auto {op_var} = builder::Squeeze({op_var}_t, {op_var}_axes);\n"

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

    @staticmethod
    def Embedding(op_var, weight, indices, *args_str):
        if args_str:
            print(f"Warning: EnflameOverrides.Embedding encounter unknown args: {args_str}, ignore it")

        collapsed_slice_dim = f"{op_var}_collapsed_slice_dim"
        embedding_dim_size = f"{op_var}_embedding_dim_size"
        slice_sizes = f"{op_var}_slice_sizes"
        offset_dims = f"{op_var}_offset_dims"
        indices_rank = f"{op_var}_indices_rank"
        collapsed_slice_dims = f"{op_var}_collapsed_slice_dims"
        start_index_map = f"{op_var}_start_index_map"
        index_vector_dim = f"{op_var}_index_vector_dim"
        gather_dim_params = f"{op_var}_gather_dim_params"
        src_code = f"int64_t {collapsed_slice_dim} = 0;\n" \
                   f"int64_t {embedding_dim_size} = {weight}.GetType().GetDimSize(1);\n" \
                   f"std::vector<int64_t> {slice_sizes} = {{1, {embedding_dim_size}}};\n" \
                   f"int64_t {indices_rank} = {indices}.GetType().GetRank();\n" \
                   f"std::vector<int64_t> {offset_dims} = {{{indices_rank}}};\n" \
                   f"std::vector<int64_t> {collapsed_slice_dims} = {{{collapsed_slice_dim}}};\n" \
                   f"std::vector<int64_t> {start_index_map} = {{0}};\n" \
                   f"int64_t {index_vector_dim} = {indices_rank};\n" \
                   f"auto {gather_dim_params} = builder::GatherDimensionNumbers(\n" \
                   f"    {offset_dims}, {collapsed_slice_dims}, {start_index_map}, {index_vector_dim}\n" \
                   f");\n" \
                   f"builder::Op {op_var} = builder::Gather(\n" \
                   f"    {weight}, {indices}, {gather_dim_params}, {slice_sizes}\n" \
                   f");\n" \

        return src_code
    
    # [a + bi] ===> tops.tuple(a, bi)
    @staticmethod
    def Complex(op_var, node, args_dict, args):
        shape = '{' + str(args[0].meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        src_code = f"std::vector<int64_t> {op_var}_in_shape{shape};\n"
        src_code += f"int {op_var}in_shape_size = {op_var}_in_shape.size();\n"
        
        src_code += f"auto {op_var}part0_limit_indices =  {op_var}_in_shape;\n"
        src_code += f"{op_var}part0_limit_indices[{op_var}in_shape_size - 1]--;\n"

        src_code += f"std::vector<int64_t> {op_var}part0_start_indices({op_var}in_shape_size, 0);\n"
        src_code += f"std::vector<int64_t> {op_var}part1_start_indices({op_var}in_shape_size, 0);\n"
        src_code += f"{op_var}part1_start_indices[{op_var}in_shape_size - 1] = 1;\n"
        src_code += f"std::vector<int64_t> {op_var}stride( {op_var}in_shape_size, 1);\n"

        src_code += f"builder::Op {op_var}_split0 = builder::Slice({args_dict[args[0].name]}, {op_var}part0_start_indices, {op_var}part0_limit_indices, {op_var}stride);\n"
        src_code += f"builder::Op {op_var}_split1 = builder::Slice({args_dict[args[0].name]}, {op_var}part1_start_indices, {op_var}_in_shape, {op_var}stride);\n"
        
        out_shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = args[0].meta['val'].dtype.__str__()
        src_code += f"builder::Type {op_var}_reshape_type({out_shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var}_tmp0 = builder::Reshape({op_var}_split0, {op_var}_reshape_type);\n"
        src_code += f"builder::Op {op_var}_tmp1 = builder::Reshape({op_var}_split1, {op_var}_reshape_type);\n"

        t = '{'
        t += f"{op_var}_tmp0, {op_var}_tmp1"
        t += '}'
        
        src_code += f"std::vector<builder::Op> {op_var}outputs{t};\n"

        src_code += f"builder::Op {op_var} = builder::Tuple({op_var}outputs);\n"
        
        return src_code

    # tops.tuple(a, bi)====>[a,b]
    @staticmethod
    def Viewasreal(op_var, node, x):
        src_code = f"int {op_var}irel = 0;\n"
        src_code += f"int {op_var}iimg = 1;\n"
        src_code += f"builder::Op {op_var}_real = builder::GetTupleElement({x}, {op_var}irel);\n"
        src_code += f"builder::Op {op_var}_imag = builder::GetTupleElement({x}, {op_var}iimg);\n"
        
        out_shape = '{' + str(list(node.meta['val'].shape)[:-1] + [1]).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        src_code += f"builder::Type {op_var}_reshape_type({out_shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var}_tmp0 = builder::Reshape({op_var}_real, {op_var}_reshape_type);\n"
        src_code += f"builder::Op {op_var}_tmp1 = builder::Reshape({op_var}_imag, {op_var}_reshape_type);\n"
        
        t = '{'
        t += f"{op_var}_tmp0, {op_var}_tmp1"
        t += '}'

        src_code += f"std::vector<builder::Op> {op_var}real_imag = {t};\n"
        dimension = len(node.meta['val'].shape)-1
        src_code += f'builder::Op {op_var} = builder::Concatenate({op_var}real_imag, {dimension});\n'

        return src_code

    #(a + bi)(c + di) = (ac -bd) + (ad + bd)i
    @staticmethod
    def Complexmul(op_var, node, x, y):
        src_code = f"int {op_var}irel = 0;\n"
        src_code += f"int {op_var}iimg = 1;\n"
        src_code += f"builder::Op {op_var}xreal = builder::GetTupleElement({x}, {op_var}irel);\n"
        src_code += f"builder::Op {op_var}ximag = builder::GetTupleElement({x}, {op_var}iimg);\n"

        src_code += f"builder::Op {op_var}yreal = builder::GetTupleElement({y}, {op_var}irel);\n"
        src_code += f"builder::Op {op_var}yimag = builder::GetTupleElement({y}, {op_var}iimg);\n"

        src_code += f"builder::Op {op_var}xreal_yreal = builder::Mul({op_var}xreal, {op_var}yreal);\n"
        src_code += f"builder::Op {op_var}ximag_yimag = builder::Mul({op_var}ximag, {op_var}yimag);\n"

        src_code += f"builder::Op {op_var}xreal_yimag = builder::Mul({op_var}xreal, {op_var}yimag);\n"
        src_code += f"builder::Op {op_var}ximag_yreal = builder::Mul({op_var}ximag, {op_var}yreal);\n"

        src_code += f"builder::Op {op_var}mul_real = builder::Sub({op_var}xreal_yreal, {op_var}ximag_yimag);\n"
        src_code += f"builder::Op {op_var}mul_imag = builder::Add({op_var}xreal_yimag, {op_var}ximag_yreal);\n"

        t = '{'
        t += f"{op_var}mul_real, {op_var}mul_imag"
        t += '}'

        src_code += f"std::vector<builder::Op> {op_var}outputs {t};\n"
        src_code += f"builder::Op {op_var} = builder::Tuple( {op_var}outputs);\n"
        
        return src_code

    @staticmethod
    def Concatenate(op_var, node, x):
        # dim in torch.cat ranges:
        # [-len(x.shape.size()), len(x.shape.size())-1]
        # handle negative dim for tops side.
        y = node.args[1]
        if (node.args[1] < 0 ):
            y = len(node.meta["val"][0].shape) + node.args[1] + 1
        return f"builder::Op {op_var} = builder::Concatenate({x}, {y});"


    @staticmethod
    def Softmax(op_var, node, x, z):
        y = node.args[1]
        return f"builder::Op {op_var} = builder::Softmax({x}, {y}, {z});"


    @staticmethod
    def Logsoftmax(op_var, node, x, z):
        y = node.args[1]
        return f"builder::Op {op_var} = builder::Softmax({x}, {y}, {z}, true);"

    @staticmethod
    def Gelu(op_var, node, x):
        y = "true"
        if not node.kwargs or ("approximate" in node.kwargs and node.kwargs["approximate"] == "none"):
            y = "false"
        return f"builder::Op {op_var} = builder::Gelu({x}, {y});"
