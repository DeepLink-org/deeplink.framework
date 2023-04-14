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
            "torch.int64": "builder::PrimitiveType::S64()"}

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
        
        in_shape = self.get_shape()
        self.build_graph_code.add_line(f'std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape};')

        data_type = type_set[str(self.cur_node.meta['val'].dtype)]
        self.build_graph_code.add_line(f'builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, {data_type});')
        self.build_graph_code.add_line(f'builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);')

    def call_function(self, name, target, args, kwargs):   
        if name not in self.args_dict.keys():
            self.args_dict[name] = 'op' + str(len(self.args_dict))

        # TODO did not handle squeeze and unsuqeeze
        arg_code, args_list = EnflameOverrides.gen_args(self.args_dict[name], self.args_dict, self.cur_node, args)
        
        op_code = getattr(self.override, target.name())(*args_list)
        self.build_graph_code.splice(arg_code)
        self.build_graph_code.splice(f'builder::Op {self.args_dict[name]} = {op_code};')
        
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
        return self.generate_code()

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
                #include "conv2d_grad.h"
                #include "max_pool2d_grad.h"

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
                from third_party.DICP.TopsGraph.compile import AsyncCompileTopsGraph
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
            func_body.add_line(f'output_ptrs.emplace_back(output_ptr{str(i)});')
        func_body.add_line(f'run(exe_ptr, input_ptrs, output_ptrs);')

        input_paras = ''
        for i in range(0, len(self.input_args)):
            input_paras += f'float* input_ptr{str(i)}, '
        output_paras = []
        for i in range(0, len(self.output_args)):
            output_paras.append(f'float* output_ptr{str(i)}')
        output_paras = ', '.join(output_paras)

        run_func_code = CodeBlock()
        run_func_code.add_line(f'extern "C" void run({input_paras} {output_paras}) {"{"}')
        run_func_code.splice(func_body)
        run_func_code.splice('}')
        return run_func_code

    def gen_tensor(self, prefix, tensor):
        shape = f"{tensor.shape}"[12:-2]
        return f"{prefix}(({shape}, ), {tensor.stride()}, device='{tensor.device.type}', dtype={tensor.dtype})"

    def gen_empty_tensor(self, tensor):
        return self.gen_tensor("empty_strided", tensor)

    def gen_random_tensor(self, tensor):
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
        args_str = []
        count = 0
        for i in range(len(args)):
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            elif isinstance(args[i], bool):
                args_str.append(str(args[i]).lower())
            elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
                args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            elif isinstance(args[i], torch.dtype):
                in_shape_size = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                src_code.add_line(f'std::vector<int64_t> {op_var}_shape{count}{in_shape_size};')
                src_code.add_line(f'builder::Type {op_var}_type{count} = builder::Type({op_var}_shape{count}, ptype);')
                args_str.append(f'{op_var}_type{count}')
                count += 1
            else:
                if flag:
                    src_code.add_line(f'builder::Type {op_var}_axes_type{count}({"{" + "1" + "}"}, builder::PrimitiveType::S64());')
                    src_code.add_line(f'std::vector<int64_t> {op_var}_axes_data{count} = {"{" + str(args[i]).split("[")[-1].split("]")[0] + "}"};')
                    src_code.add_line(f'builder::Op {op_var}_axes{count} = builder::Const(hlir_builder, ({op_var}_axes{count}_data.data()), {op_var}_axes_type{count});')
                    args_str.append(f'{op_var}_axes{count}')
                    shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                    src_code.add_line(f"builder::Type {op_var}_output_type{count}({shape}, builder::PrimitiveType::F32());")
                    args_str.append(f"{op_var}_output_type{count}")
                    count += 1
                else:
                    in_shape_size = '{1}'
                    if isinstance(type(args[i]), type(int)):
                        src_code.add_line(f'int {op_var}_value{count} = {str(args[i])};')
                    else:
                        src_code.add_line(f'float {op_var}_value{count} = {str(args[i])};')
                    src_code.add_line(f'std::vector<int64_t> {op_var}_const_in_shape{count}{in_shape_size};')
                    src_code.add_line(f'builder::Type {op_var}_value_type{count}({op_var}_const_in_shape{count}, ptype);')
                    src_code.add_line(f'builder::Op {op_var}_const{count} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_value{count}), {op_var}_value_type{count});');
                    args_str.append(f'{op_var}_const{count}')
                    count += 1

        return src_code, args_str

    @staticmethod
    def abs(x):
        return f'builder::Abs({x})'

    @staticmethod
    def reciprocal(x):
        return f'builder::Reciprocal({x})'

    @staticmethod
    def add(x, y):
        return f'builder::Add({x}, {y})'
 
    @staticmethod
    def sub(x, y):
        return f'builder::Sub({x}, {y})'
    
    @staticmethod
    def mul(x, y):
        return f'builder::Mul({x}, {y})'
    
    @staticmethod
    def div(x, y):
        return f'builder::Div({x}, {y})'
    
    @staticmethod
    def log(args_dict, node, args):
        return self.gen_opcode('Log', args_dict, node, args)
    
    @staticmethod
    def neg(args_dict, node, args):
        return self.gen_opcode('Neg', args_dict, node, args)
    
    @staticmethod
    def exp(args_dict, node, args):
        return self.gen_opcode('Exp', args_dict, node, args)

    @staticmethod
    def sqrt(x):
        return f'builder::Sqrt({x})'
     
    @staticmethod
    def relu(args_dict, node, args):
        return self.gen_opcode('Relu', args_dict, node, args)
    
    @staticmethod
    def lessequal(args_dict, node, args):
        return self.gen_opcode('LessEqual', args_dict, node, args)
    
    @staticmethod
    def squeeze(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args, True)
        src_code += f"  builder::Op {args_dict[node.name]} = builder::Squeeze({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def unsqueeze(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args, True)
        src_code += f"  builder::Op {args_dict[node.name]} = builder::Unsqueeze({', '.join(args_str)});\n\n"
        return src_code
        
    @staticmethod
    def clone(args_dict, node, args):
        return f"  builder::Op {args_dict[node.name]} = {args_dict[args[0].name]};\n"
    
    @staticmethod
    def reducemean(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
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
    def reshape(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code += f'  builder::Type reshape_shape{EnflameOverrides.count}({shape}, ptype);\n'
        args_str[1] = f'reshape_shape{EnflameOverrides.count}'
        EnflameOverrides.count += 1
        
        src_code += f"  builder::Op {args_dict[node.name]} = builder::Reshape({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def addmm(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
        src_code = f"  builder::Op addmm{EnflameOverrides.count} = builder::Gemm({'{' + args_str[1] + ', ' + args_str[2] + '}'});\n"
        src_code += f"  builder::Op {args_dict[node.name]} = builder::Add({args_str[0]}, addmm{EnflameOverrides.count});\n"
        EnflameOverrides.count += 1
        return src_code
    
    @staticmethod
    def reduceMax(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
        if len(args_str) ==3:
            args_str[1], args_str[2] = args_str[2], args_str[1]
        src_code = f"  builder::Op {args_dict[node.name]} = builder::ReduceMax({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def ReduceSum(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
        if len(args_str) ==3:
            args_str[1], args_str[2] = args_str[2], args_str[1]
        src_code = f"  builder::Op {args_dict[node.name]} = builder::ReduceSum({', '.join(args_str)});\n\n"
        return src_code   
    
    @staticmethod
    def Gemm(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
        src_code = f"  builder::Op {args_dict[node.name]} = builder::Gemm({'{' + ', '.join(args_str) + '}'});\n\n"
        return src_code

    @staticmethod
    def Transpose(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
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
        src_code += f"  builder::Op {args_dict[node.name]} = builder::GetTupleElement({args_dict[args[0].name]}, {int(node.args[1])}, getitem_type{EnflameOverrides.count});\n\n"
        EnflameOverrides.count += 1
        return src_code
    
    @staticmethod
    def Gather(args_dict, node, args):
        src_code, args_str = self.gen_args(args_dict, node, args)
        
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
        src_code, args_str = self.gen_args(args_dict, node, args)    

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
        # print("args_strlen:", len(args_str))
        src_code += f"  auto {args_dict[node.name]} = enflame::batch_norm(hlir_builder, {args_str[0]}, {args_str[1]}, {args_str[2]});\n"     
        EnflameOverrides.count += 1
        
        return src_code
    
    @staticmethod
    def Threshold_Backward(args_dict, node, args):
        src_code = f"  builder::Op {args_dict[node.name]} = builder::ReluGrad({', '.join(args_str)});\n\n"
        return src_code
    
    @staticmethod
    def Convolution(args_dict, node, args):
        args_str =[]
        for i in range(0, 3):
            if isinstance(args[i], type(None)):
                continue
            args_str.append(args_dict[args[i].name])
                
        stride = str(args[3]).replace('[','{').replace(']','}')
        
        if len(args[4]) == 1:
            row_padding, col_padding = args[4][0]
        else:
            row_padding = args[4][0]
            col_padding = args[4][1]
        
        padding = f"{'{' + str(row_padding)}, {str(row_padding)}, {str(col_padding)}, {str(col_padding) + '}'}"
        
        dilation = str(args[5]).replace('[','{').replace(']','}')
        group = args[8]
    
        src_code = f"  std::vector<builder::Op> {args_dict[node.name]}_inputs = {'{' + ', '.join(args_str) + '}'};\n"
        src_code += f'  builder::Op {args_dict[node.name]} = builder::Conv2D({args_dict[node.name]}_inputs, {group}, "NOTSET", "NCHW", {stride}, {padding}, {dilation});\n\n'
        
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
