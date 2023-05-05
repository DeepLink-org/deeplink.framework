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

import torch

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate

from torch._inductor.codegen.common import OpOverrides


type_set = {"torch.float32": "builder::PrimitiveType::F32()",
            "torch.int64": "builder::PrimitiveType::S64()",
            "torch.bool": "builder::PrimitiveType::PRED()"}

need_node = ['reshape', 'Getitem', 'Gather', 'Batch_Norm', 'Convolution', 'Conv2D_Grad',
             'MaxPool2D', 'MaxPool2D_Grad', 'Zeros', 'Expand']

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

    return real_op

class CodeBlock:
    tabwidth = 4

    def __init__(self, backend='', indent=0, lang=''):
        self._lines = []
        self._indent = indent
        self._backend = backend
        self._lang = lang

    def get_str(
        self,
    ):
        buf = StringIO()
        for line in self._lines:
            assert isinstance(line, str)
            buf.write(line)
            if self._lang == 'cpp' or self._lang == 'c':
                buf.write(";")
            buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def add_line(self, line):
        if line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def add_lines(self, lines):
        for line in lines:
            self.add_line(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            yield
            self._indent -= offset

        return ctx()

    def splice(self, other_code, dedent=False):
        if isinstance(other_code, CodeBlock):
            _dedent = float("inf")
            if dedent:
                for line in other_code._lines:
                    if line:
                        _dedent = min(_dedent, len(line) - len(line.lstrip()))
            if math.isinf(_dedent):
                _dedent = 0
            for line in other_code._lines:
                CodeBlock.add_line(self, line[_dedent:])
        else:
            if dedent:
                other_code = textwrap.dedent(other_code)
            other_code = other_code.lstrip()
            other_code = other_code.rstrip()
            if not other_code:
                return
            for line in other_code.split("\n"):
                self.add_line(line)

class EnflameCodegen(torch.fx.Interpreter):
    def __init__(self, graph):
        self.name = 'topsgraph'
        self.import_code = CodeBlock()
        
        self.args_dict = {}
        self.input_args =[]
        self.output_args = []
        self.build_graph_code = CodeBlock(indent=1, lang='cpp')
        
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
        self.build_graph_code.add_line(f'std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape};')

        self.build_graph_code.add_line(f'builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, {type_set[data_type]});')
        self.build_graph_code.add_line(f'builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);')

    def call_function(self, name, target, args, kwargs):   
        if name not in self.args_dict.keys():
            self.args_dict[name] = 'op' + str(len(self.args_dict))

        # TODO did not handle squeeze and unsuqeeze
        arg_code, args_list = EnflameOverrides.gen_args(self.args_dict[name], self.args_dict, self.cur_node, args)
        real_op = process_name(name, target)
        
        print("*******************************************", flush=True)
        print("name:", name, flush=True)
        print("target:", target.name(), flush=True)
        print("real_op:", real_op, flush=True)
        print("args:", args, flush=True)
        print("arg_code:", arg_code.get_str(), flush=True)
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
        with open('codegen1.py', 'w') as f:
            f.write(test)
        return test

    def gen_build_graph_code(self):
        graph_code = CodeBlock(indent=1)
        graph_code.add_lines(
            [
                f'auto hlir_builder = std::make_shared<builder::Builder>();',
                f'hlir_builder->SetShapeInference(true);',
                f'auto ptype = builder::PrimitiveType::F32();'
            ]
        )

        graph_code.splice(self.build_graph_code, dedent=True)
        
        output_str = []
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                continue
            else:
                output_str.append(self.args_dict[self.output_args[i].name])

        graph_code.add_line(f'hlir_builder->SetOutput({"{" + ", ".join(output_str) + "}"});')

        graph_code.add_line(f'return hlir_builder;')
        return graph_code

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
            , dedent=True
        )
        return self.import_code.get_str()

    def gen_compile_func_code(self):
        compile_func_body = CodeBlock(indent=1)
        compile_func_body.splice(
            f"""
              auto hlir_builder = build_sample();
                compile(hlir_builder, &exe_ptr);
            """
            , dedent=True
        )
        compile_func = CodeBlock()
        compile_func.add_lines(
            [
                f'topsExecutable_t exe_ptr;',
                f'extern "C" void compile(void){"{"}'
            ]
        )
        compile_func.splice(compile_func_body)
        compile_func.add_line('}')

        return compile_func
                
    def gen_run_func_code(self):
        func_body = CodeBlock(indent=1)
        func_body.add_line(f'std::vector<void *> input_ptrs;')
        for i in range(0, len(self.input_args)):
            func_body.add_line(f'input_ptrs.emplace_back(static_cast<void *>(input_ptr{str(i)}));')
        func_body.add_line(f'std::vector<void *> output_ptrs;')
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                func_body.add_line(f'output_ptrs.emplace_back(output_ptr{str(i)});')
        func_body.add_line(f'run(exe_ptr, input_ptrs, output_ptrs);')

        input_paras = ''
        for i in range(0, len(self.input_args)):
            input_paras += f'float* input_ptr{str(i)}, '
        output_paras = []
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                output_paras.append(f'float* output_ptr{str(i)}')
        output_paras = ', '.join(output_paras)

        run_func_code = CodeBlock()
        run_func_code.add_line(f'extern "C" void run({input_paras} {output_paras}) {"{"}')
        run_func_code.splice(func_body)
        run_func_code.splice('}')
        return run_func_code

    def gen_tensor(self, prefix, tensor):
        #shape = f"{tensor.shape}"[12:-2]
        #print(f"{'!' * 20}  temsor:\n {tensor}\n shape :\n {shape}")
        #if shape == '':
        #    return f"{prefix}((), (), device='{tensor.device.type}', dtype={tensor.dtype})"
        res =  f"{prefix}({tuple(tensor.shape)}, {tensor.stride()}, device='{tensor.device.type}', dtype={tensor.dtype})"
        #print(f"{'!' * 20}  res:\n {res}\n")
        return res

    def gen_empty_tensor(self, tensor):
        return self.gen_tensor("empty_strided", tensor)

    def gen_random_tensor(self, tensor):
        # tmp = self.gen_tensor("rand_strided", tensor)
        # tmp = tmp.replace('rand_strided((), (),', 'rand_strided((1,), (1,),')
        return self.gen_tensor("rand_strided", tensor)

    def gen_call_func(self):
        # TODO check scalar input
        call_body = CodeBlock(indent=1)

        args = []
        for i in range(len(self.input_args)):
            args.append('arg' + str(i))
        call_body.add_line(f"{', '.join(args)}, = args")
        call_body.add_line(f"args.clear()")

        bufs = []
        for i in range(len(self.output_args)):
            bufs.append('buf' + str(i))
            if isinstance(self.output_args[i], type(None)):
                call_body.add_line(bufs[-1] + ' = ' + (f"empty_strided((), ())"))
            else:
                otensor = self.output_args[i].meta['val']
                call_body.add_line(bufs[-1] + ' = ' + self.gen_empty_tensor(otensor))

        call_str = 'kernel_cpp_0('
        for i in range(len(self.input_args)):
            call_str += 'c_void_p(arg' + str(i) + '.data_ptr()), '
        for i in range(len(self.output_args)):
            call_str += 'c_void_p(' + bufs[i] + '.data_ptr())'
            if i != len(self.output_args) - 1:
                call_str += ', '
            else:
                call_str += ')'
        call_body.add_line(call_str)

        for arg in args:
            call_body.add_line(f'del {arg}')
        
        call_body.add_line(f"return ({', '.join(bufs)})")

        call_func = CodeBlock()
        call_func.add_line(f"def call(args):")
        call_func.splice(call_body)
 
        return call_func.get_str()

    def gen_main_func(self):
        main_body = CodeBlock(indent=1)
        main_body.splice(
            f"""
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
            """
            , dedent=True
        )
        for i in range(0, len(self.input_args)):
            itensor = self.input_args[i].meta['val']
            main_body.add_line('arg' + str(i) + ' = ' + self.gen_random_tensor(itensor))

        args = []
        for i in range(len(self.input_args)):
            args.append('arg' + str(i))
        main_body.add_line(f"print(call([{', '.join(args)}]))")

        main_func = CodeBlock()
        main_func.add_line(f"""if __name__ == "__main__":""")
        main_func.splice(main_body)
        return main_func.get_str()

    def gen_compile_graph_code(self):
        compile_graph_code = CodeBlock()
        compile_graph_code.splice(
            f"""
                async_compile = AsyncCompileTopsGraph()
                kernel_cpp_0 = async_compile.topsgraph('''
            """
            , dedent=True
        )
        compile_graph_code.splice(self.get_kernel_header(), dedent=True)
        compile_graph_code.add_line(f'std::shared_ptr<builder::Builder> build_sample() {"{"}')
        compile_graph_code.splice(self.gen_build_graph_code())
        compile_graph_code.add_line('}')

        compile_graph_code.splice(self.gen_compile_func_code())
        compile_graph_code.splice(self.gen_run_func_code())
        compile_graph_code.add_line(f"''')")
        compile_graph_code.add_line('async_compile.wait(globals())')
        compile_graph_code.add_line('del async_compile')

        return compile_graph_code.get_str()

    def generate_code(self):
        return (self.gen_import_code() + self.gen_compile_graph_code()+ self.gen_call_func() + self.gen_main_func())

class EnflameOverrides(OpOverrides):
    @staticmethod
    def gen_args(op_var, args_dict, node, args, flag=False):
        src_code = CodeBlock()
        args_str = [op_var]
        count = 0
        if process_name(node.name, node.target) in need_node:
            print('process_name(node.name, node.target):', process_name(node.name, node.target), flush=True)
            args_str.append(node)

        for i in range(len(args)):
            if isinstance(args[i], type(None)):
                continue
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            elif isinstance(args[i], bool):
                args_str.append(str(args[i]).lower())
            elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
                if node.name == "reducemean" and len(args) != 1 and i == 1:
                    tmp = args[1].copy()
                    tmp.sort()
                    for j in range(0, len(tmp)):
                        tmp[j] = (tmp[j] + len(args[0].meta['val'].shape)) % len(args[0].meta['val'].shape)
                    args_str.append(str(tmp).replace('[', '{').replace(']', '}'))
                else:
                    args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            elif isinstance(args[i], torch.dtype):
                in_shape_size = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                src_code.add_line(f'std::vector<int64_t> {op_var}_shape{count}{in_shape_size};')
                src_code.add_line(f'builder::Type {op_var}_type{count} = builder::Type({op_var}_shape{count}, ptype);')
                args_str.append(f'{op_var}_type{count}')
                count += 1
            else:
                if node.name == "squeeze" or node.name == "unsqueeze":
                    src_code.add_line(f'builder::Type {op_var}_axes_type{count}({"{" + "1" + "}"}, builder::PrimitiveType::S64());')
                    if node.name == "squeeze":
                        axes = args[i]
                        src_code.add_line(f'std::vector<int64_t> {op_var}_axes_data{count} = {"{" + str(axes) + "}"};\n')
                    elif node.name == "unsqueeze":
                        src_code.add_line(f'std::vector<int64_t> {op_var}_axes_data{count} = {"{" + str(args[i]).split("[")[-1].split("]")[0] + "}"};')

                    src_code.add_line(f'builder::Op {op_var}_axes{count} = builder::Const(hlir_builder, ({op_var}_axes_data{count}.data()), {op_var}_axes_type{count});')
                    args_str.append(f'{op_var}_axes{count}')
                    shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                    data_type = node.meta['val'].dtype.__str__()
                    if data_type == 'torch.float32':
                        out_type = 'builder::PrimitiveType::F32()'
                    elif data_type == 'torch.int64':
                        out_type = 'builder::PrimitiveType::S64()'
                    else:
                        raise ValueError("Unknown data type.")

                    src_code.add_line(f"builder::Type {op_var}_output_type{count}({shape}, {out_type});\n")
                    args_str.append(f"{op_var}_output_type{count}")
                    count += 1

                elif node.name == "Scatter":
                    if i  == 1:
                        args_str.append(str(args[1]))
                    else:
                        if isinstance(args[3], float):
                            src_code.add_line(f'  const float {op_var}_value{count} = {str(args[3])};\n')
                        else:
                            src_code.add_line(f'  const int {op_var}_value{count} = {str(args[3])};\n')
                            args_str.append(f'{op_var}_value{count}')
                else:
                    in_shape_size = '{1}'
                    print(type(node.meta['val']))
                    if isinstance(node.meta['val'], list) or isinstance(node.meta['val'], tuple):
                        data_type = node.meta['val'][0].dtype.__str__()
                    else:
                        data_type = node.meta['val'].dtype.__str__()
                    if data_type == 'torch.int64':
                        out_type = 'builder::PrimitiveType::S64()'
                        src_code.add_line(f'int {op_var}_value{count} = {str(args[i])};\n')
                    else:
                        out_type = 'builder::PrimitiveType::F32()'
                        src_code.add_line(f'float {op_var}_value{count} = {str(args[i])};\n')
  
                    src_code.add_line(f'std::vector<int64_t> {op_var}_const_in_shape{count}{in_shape_size};')
                    src_code.add_line(f'builder::Type {op_var}_value_type{count}({op_var}_const_in_shape{count}, {out_type});')
                    src_code.add_line(f'builder::Op {op_var}_const{count} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_value{count}), {op_var}_value_type{count});')
                    args_str.append(f'{op_var}_const{count}')
                    count += 1

        return src_code, args_str

    @staticmethod
    def abs(op_var, x):
        return f"builder::Op {op_var} = builder::Abs({x});"

    @staticmethod
    def reciprocal(x):
        return f'builder::Reciprocal({x})'

    @staticmethod
    def add(op_var, x, y):
        return f"builder::Op {op_var} = builder::Add({x}, {y});"
 
    @staticmethod
    def sub(op_var, x, y):
        return f"builder::Op {op_var} = builder::Sub({x}, {y});"
    
    @staticmethod
    def mul(op_var, x, y):
        return f"builder::Op {op_var} = builder::Mul({x}, {y});"
    
    @staticmethod
    def div(op_var, x, y):
        return f"builder::Op {op_var} = builder::Div({x}, {y});"
    
    @staticmethod
    def log(op_var, x):
        return f"builder::Op {op_var} = builder::Log({x});"
    
    @staticmethod
    def neg(op_var, x):
        return f"builder::Op {op_var} = builder::Neg({x});"
    
    @staticmethod
    def exp(op_var, x):
        return f"builder::Op {op_var} = builder::Exp({x});"

    @staticmethod
    def sqrt(op_var, x):
        return f"builder::Op {op_var} = builder::Sqrt({x});"
     
    @staticmethod
    def relu(op_var, x):
        return f"builder::Op {op_var} = builder::Relu({x});"
    
    @staticmethod
    def lessequal(op_var, x, y):
        return f'builder::Op {op_var} = builder::LessEqual({x}, {y});'
    
    @staticmethod
    def squeeze(op_var, *args):
        return f"builder::Op {op_var} = builder::Squeeze({', '.join(args)});"
    
    @staticmethod
    def unsqueeze(op_var, *args):
        return f"builder::Op {op_var} = builder::Unsqueeze({', '.join(args)});"

    @staticmethod
    def clone(op_var, x):
        return f"builder::Op {op_var} = {x};"

    # TODO need node
    @staticmethod
    def reducemean(op_var, *args_str):
        if len(args_str) == 3:
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args_str[0]}, {args_str[2]}, {args_str[1]});\n"
        elif len(args_str) == 2:
            keepdim = 'false'
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args_str[0]}, {keepdim}, {args_str[1]});\n"
        elif len(args_str) == 1:
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args_str[0]});\n"
        else:   
            ValueError("Reducemean args num error!")
        return src_code

    @staticmethod
    def reshape(op_var, node, *args):
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f'builder::Type _reshape_shape{op_var}({shape}, ptype);\n'
        tmp = f'_reshape_shape{op_var}'

        src_code += f"builder::Op {op_var} = builder::Reshape({args[0]}, {tmp});\n\n"
        return src_code

    @staticmethod
    def ReduceMax(op_var, *args):
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
    def ReduceSum(op_var, *args):
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
    def Gemm(op_var, x, y):
        return f"builder::Op {op_var} = builder::Gemm({'{' + x + ',' + y + '}'});"

    # TODO need test
    @staticmethod
    def Select(op_var, *args):
        return f"builder::Op {op_var} = builder::Select({', '.join(args)});"

    @staticmethod
    def transpose(op_var, *args):
        if len(args) == 1:
            args.append('{1, 0}')
        return f"builder::Op {op_var} = builder::Transpose({', '.join(args)});"

    # TODO need node
    @staticmethod
    def Getitem(op_var, node, *args):
        src_code = f"builder::Op {op_var} = builder::GetTupleElement({args[0]}, {int(node.args[1])});\n\n"
        return src_code

    # TODO need node
    @staticmethod
    def Gather(op_var, node, *args_str):
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        new_args_str = []
        new_args_str.append(args_str[0])
        new_args_str.append(args_str[2])
        new_args_str.append(str(node.args[1]))
        
        count_num = node.name.split('_')
        if len(count_num) == 1:
            count_num = 0
        else:
            count_num = int(count_num[1])
        
        src_code = f"builder::Type gather_type{count_num}({shape}, builder::PrimitiveType::F32());\n"
        new_args_str.append(f"gather_type{count_num}")
        
        src_code += f"auto {op_var} = enflame::Gather(hlir_builder, {', '.join(new_args_str)});\n"

        return src_code

    # TODO need node
    @staticmethod
    def Batch_Norm(op_var, node, *args_str):
        shape = []
        lenth = len(node.meta['val'])
        for i in range (0, lenth):
            shape.append('{' + str(node.meta['val'][i].shape).split('[')[-1].split(']')[0] + '}')

        src_code = f"std::vector<std::vector<int64_t>> tuple_shape{op_var};\n"
        src_code += f"std::vector<builder::PrimitiveType> tuple_dtype{op_var};\n"

        src_code += f"builder::Type bn_type{op_var}(tuple_shape{op_var}, tuple_dtype{op_var});\n"
        args_str_tmp = args_str[:5]
        src_code += f"auto {op_var} = enflame::BatchNorm(hlir_builder, {', '.join(args_str_tmp)}, 1, true, {str(node.args[6])}, {str(node.args[7])});\n"     

        return src_code
    
    @staticmethod
    def Threshold_Backward(op_var, *args_str):
        src_code = f"builder::Op {op_var} = builder::ReluGrad({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Convolution(op_var, node, *args_str):
        tmp_str =[]
        index = 3
        # print("Convolution args_str:", args_str, flush=True)
        for i in range(0, 3):
            if isinstance(node.args[i], type(None)):
                index -= 1
                continue
            tmp_str.append(args_str[i])
        
        # print("index", index, flush=True)
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

    # TODO did not test
    @staticmethod
    def Conv2D_Grad(op_var, node, *args_str):
        new_args_str = []
        for i in range(0, 3):
            if isinstance(node.args[i], type(None)):
                continue
            else:
                new_args_str.append(args_str[i])

        bias = args_str[3]
        stride = args_str[4]
        padding = args_str[5]
        dilation = args_str[6]
        
        src_code = f"auto {op_var} = enflame::Conv2D_Grad(hlir_builder, {','.join(new_args_str)}, {bias}, {stride}, {padding}, {dilation});\n"

        return src_code

    # TODO need node
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
        
        src_code += f"auto {op_var} = enflame::MaxPool2D_Grad(hlir_builder, {args_str[0]}, {args_str[2]}, {ksize}, {strides}, {padding});\n"
        
        return src_code

    @staticmethod
    def Reciprocal(op_var, *args):
        return f"builder::Op {op_var} = builder::Reciprocal({', '.join(args)});"

    # TODO no test
    @staticmethod
    def Zeros(op_var, node, *args):
        data_type = node.meta['val'].dtype.__str__()
        if data_type == 'torch.float32':
            out_type = 'builder::PrimitiveType::F32()'
        elif data_type == 'torch.int64':
            out_type = 'builder::PrimitiveType::S64()'
        else:
            raise ValueError("Data type error!")
            
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        src_code = f"builder::Op {op_var} = builder::ZerosLike({args[0]}, {out_type}, {shape});\n\n"

        return src_code

    # TODO need test
    @staticmethod
    def Scatter(op_var, *args):
        src_code = f"auto {op_var} = enflame::Scatter(hlir_builder, {', '.join(args)});\n"
        return src_code

    # TODO need test 
    @staticmethod
    def Convert(op_var, *args):
        return f"builder::Op {op_var} = builder::Convert({', '.join(args)});"

    # TODO need test
    @staticmethod
    def Scalar(op_var, *args):
        shape = '{1}'
        src_code = f"builder::Type scalar_type{op_var}({shape}, builder::PrimitiveType::F32());"
        src_code += f"std::vector<float> scalar_data{op_var}(1, {args[0]});\n"
        src_code += f"auto {op_var} = builder::Const(hlir_builder, static_cast<void *>(scalar_data{op_var}.data()), scalar_type{op_var});\n"

        return src_code

    # TODO need test
    @staticmethod
    def Expand(op_var, node, *args):
        dims = []
        for i in range(0, len(node.meta['val'].shape)):
            dims.append(str(i))
        broadcast_dims = '{' + ', '.join(dims) + '}'

        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        data_type = node.meta['val'].dtype.__str__()
        if data_type == 'torch.float32':
            out_type = 'builder::PrimitiveType::F32()'
        elif data_type == 'torch.int64':
            out_type = 'builder::PrimitiveType::S64()'
            
        src_code = f'builder::Type expand_type{op_var}({shape}, {out_type});\n'
        # TODO
        src_code += f"auto {op_var} = enflame::BroadcastInDim(hlir_builder, {args[0]}, {broadcast_dims}, expand_type{op_var});\n"
        return src_code