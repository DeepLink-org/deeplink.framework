import contextlib
import dataclasses
import functools
import math
import sys
import os
import numbers
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
from ..config import tops_debug, dipu_flag, tops_check_precision


type_set = {torch.float16: "builder::PrimitiveType::F16()",
            torch.half: "builder::PrimitiveType::F16()",
            torch.float32: "builder::PrimitiveType::F32()",
            torch.float: "builder::PrimitiveType::F32()",
            torch.float64: "builder::PrimitiveType::F64()",
            torch.double: "builder::PrimitiveType::F64()",
            torch.int8: "builder::PrimitiveType::S8()",
            torch.int16: "builder::PrimitiveType::S16()",
            torch.short: "builder::PrimitiveType::S16()",
            torch.int32: "builder::PrimitiveType::S32()",
            torch.int: "builder::PrimitiveType::S32()",
            torch.int64: "builder::PrimitiveType::S64()",
            torch.long: "builder::PrimitiveType::S64()",
            torch.uint8: "builder::PrimitiveType::U8()",
            torch.bool: "builder::PrimitiveType::PRED()",
            torch.complex64: "builder::PrimitiveType::F32()"}

cxx_type_set = {torch.float32: "float_t",
                torch.float: "float_t",
                torch.float64: "double_t",
                torch.double: "double_t",
                torch.int32: "int32_t",
                torch.int: "int32_t",
                torch.int64: "int64_t",
                torch.long: "int64_t",
                torch.bool: "bool"}

need_node = ['Scalar', 'Div', 'ReduceSum', 'Reshape', 'Expand', 'ZerosLike', 'EmptyLike', 'Bernoulli', 'OnesLike', 'Full', 'FullLike', 'Getitem', 'Gather', 'Scatter',
             'Batch_Norm', 'Convolution', 'Conv2D_Grad', 'MaxPool2D', 'MaxPool2D_Grad', 'AvgPool2D_Grad', 'Complex', 'Bmm', 'Slice', 'Select', 
             'Viewasreal', 'Complexmul', 'Concatenate', 'Gelu', 'Gelu_Grad', 'Iota', 'NativeDropout', 'Index', 
             'ArangeDefault', 'SliceScatter']

need_dict = ['Div', 'Bmm', 'Slice', 'Select', 'Complex', 'Concatenate']

not_gen_const = ['Scalar', 'Reshape', 'Expand', 'ZerosLike', 'EmptyLike', 'Bernoulli', 'OnesLike', 'Full', 'FullLike', 'Getitem', 'Gather', 'Slice', 'Scatter', 
                 'Batch_Norm', 'Convolution', 'Conv2D_Grad', 'MaxPool2D', 'MaxPool2D_Grad', 'Complex', 'Viewasreal', 'Complexmul', 
                 'Concatenate', 'Softmax', 'Logsoftmax', 'Gelu', 'Gelu_Grad', 'Iota', 'NativeDropout', 'ArangeDefault', 'SliceScatter']

def process_name(name, target):
    if hasattr(target, "name"):
        real_op = target.name().split('::')[-1]
        if real_op.find('.') != -1:
            real_op = real_op.split('.')[0]
    else:
        real_op = name.rsplit('_', 1)[0] if name[-1].isdigit() else name

    return real_op

class EnflameCodegen(torch.fx.Interpreter):
    def __init__(self, graph, origin_graph=None, folder=None, graph_key=None):
        self.name = 'topsgraph'
        self.device_id = os.getenv('DICP_TOPS_DEVICE_ID', default='0')
        
        self.import_code = IndentedBuffer()
        
        self.args_dict = {}
        self.input_args =[]
        self.output_args = []
        self.build_graph_code = IndentedBuffer(initial_indent=1)
        
        self.graph = graph
        self.folder = folder
        self.graph_key = graph_key

        super().__init__(graph)
        self.override = EnflameOverrides

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = 'op' + str(len(self.args_dict))
        self.input_args.append(self.cur_node)
        
        if dipu_flag:
            self.device_id = self.cur_node.meta['val'].device.index
            
        data_type = self.cur_node.meta['val'].dtype
        if data_type not in type_set.keys():
            print("data_type:", data_type, flush=True)
            raise ValueError("Type error!")
    
        in_shape = self.get_shape()
        if in_shape == '{}':
            in_shape = '{1}'
        self.build_graph_code.writeline(f"std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape};")
        self.build_graph_code.writeline(f"builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, {type_set[data_type]});")
        self.build_graph_code.writeline(f"builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);")
        self.build_graph_code.writeline("")

    def get_attr(self, name, target, args, kwargs):
        assert isinstance(target, str)
        if name not in self.args_dict.keys():
            op_var = self.args_dict[name] = name + '_' + str(len(self.args_dict))
        attr = self.fetch_attr(target)
        assert(isinstance(attr, torch.Tensor))
        if attr.size():
            data = str(attr)[str(attr).find('(') + 1 : -1].replace('[', '{').replace(']', '}')
            shape = '{' + str(attr.shape).split('[')[-1].split(']')[0] + '}'
            self.build_graph_code.writeline(f"builder::Type {op_var}_type({shape}, {type_set[attr.dtype]});")
            self.build_graph_code.writeline(f"std::vector<{cxx_type_set[attr.dtype]}> {op_var}_data{data};")
            self.build_graph_code.writeline(f"builder::Op {op_var} = builder::Const(hlir_builder, {op_var}_data.data(), {op_var}_type);")
        else:
            self.build_graph_code.writeline(f"builder::Type {op_var}_type({{1}}, {type_set[attr.dtype]});")
            self.build_graph_code.writeline(f"builder::Op {op_var} = builder::Const(hlir_builder, {attr}, {op_var}_type);")
        
    def call_function(self, name, target, args, kwargs):
        if name not in self.args_dict.keys():
            op_var = self.args_dict[name] = name + '_' + str(len(self.args_dict))

        arg_code, args_list, kwargs_list = EnflameOverrides.gen_args(self.args_dict, args, kwargs)
        real_op = process_name(name, target)
        
        if tops_debug:
            print("*******************Debug info*******************")
            print("name:", name)
            print("target:", target.name())
            print("real_op:", real_op)
            print("args:", args)
            print("arg_code:", arg_code.getvalue())
            print("args_list:", args_list)
            print("kwargs_list:", kwargs_list)

        #op_code = getattr(self.override, real_op)(*args_list)
        node_shape = None if not hasattr(self.cur_node, 'meta') else self.cur_node.meta['val'].shape
        node_dtype = None if not hasattr(self.cur_node, 'meta') else self.cur_node.meta['val'].dtype
        op_code = getattr(self.override, real_op)(op_var, node_shape, node_dtype, *args_list, **kwargs_list)
        
        self.build_graph_code.splice(arg_code)
        self.build_graph_code.splice(op_code)
        self.build_graph_code.writeline("")
        
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
            print(test, flush=True)
            
        return test

    def get_shape(self):
        shape = '{' + str(self.cur_node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        return shape

    def gen_import_code(self):
        self.import_code.splice(
            f"""
                import torch
                {"import torch_dipu" if dipu_flag else ""}
                
                from ctypes import c_void_p, c_long
                from torch import empty_strided, as_strided, device
                from dicp.dynamo_bridge.compile import AsyncCompileKernel
                from dicp.vendor.TopsGraph.compile_job import TopsCompileJob
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
                f'auto f16_type = builder::PrimitiveType::F16();',
                f'auto f32_type = builder::PrimitiveType::F32();',
                f'auto s64_type = builder::PrimitiveType::S64();'
            ]
        )
        graph_code.writelines("")
        graph_code.splice(self.build_graph_code, strip=True)
        
        output_str = []
        for i in range(0, len(self.output_args)):
            if isinstance(self.output_args[i], type(None)):
                continue
            else:
                output_str.append(self.args_dict[self.output_args[i].name])

        graph_code.writeline("")
        graph_code.writeline(f'hlir_builder->SetOutput({"{" + ", ".join(output_str) + "}"});')
        graph_code.writeline("")
        graph_code.writeline(f'return hlir_builder;')
        return graph_code
    
    def gen_compile_func_code(self):
        compile_func_body = IndentedBuffer()
        with compile_func_body.indent():
            compile_func_body.splice(
                f"""
                    auto hlir_builder = build_sample();
                    compile(hlir_builder, &exe_ptr, compile_bin_path);
                """
                , strip=True
            )
        compile_func = IndentedBuffer()
        compile_func.writelines(
            [
                f'topsExecutable_t exe_ptr;\n',
                f'extern "C" void compile_out(const wchar_t *compile_bin_path){"{"}'
            ]
        )
        with compile_func.indent():
            compile_func.splice(compile_func_body)
        compile_func.writeline("}")
        compile_func.writeline("")

        return compile_func
                
    def gen_run_func_code(self):
        func_body = IndentedBuffer()
        func_body.writeline(f'std::vector<void *> input_ptrs;')
        for i in range(0, len(self.input_args)):
            func_body.writeline(f'input_ptrs.emplace_back(static_cast<void *>(input_ptr{str(i)}));')
        
        func_body.writeline("")
        func_body.writeline(f'std::vector<void *> output_ptrs;')
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                func_body.writeline(f'output_ptrs.emplace_back(output_ptr{str(i)});')

        func_body.writeline("")
        func_body.writeline(f'run(exe_ptr, dipu_stream, input_ptrs, output_ptrs, {self.device_id}, {"true" if dipu_flag else "false"});')

        input_paras = ''
        for i in range(0, len(self.input_args)):
            input_paras += f'float* input_ptr{str(i)}, '
        output_paras = []
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                output_paras.append(f'float* output_ptr{str(i)}')
        output_paras = ', '.join(output_paras)

        run_func_code = IndentedBuffer()
        run_func_code.writeline(f'extern "C" void run(void *dipu_stream, {input_paras} {output_paras}) {"{"}')
        with run_func_code.indent():
            run_func_code.splice(func_body)
        run_func_code.splice('}')
        return run_func_code
    
    def gen_load_func_code(self):
        func_body = IndentedBuffer()
        func_body.writeline(f'load(&exe_ptr, compile_bin_path);')

        run_func_code = IndentedBuffer()
        run_func_code.writeline(f'extern "C" void load(const wchar_t *compile_bin_path){"{"}')
        
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
                """

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        compile_graph_code.writeline("")
        if tops_check_precision:
            compile_graph_code.splice(
                f"""
                    def check_res(a, b, graph_name):
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            check_flag = torch.allclose(a, b, atol=1e-03, equal_nan=True)
                            if check_flag:
                                print(f"cpu test success!({'{graph_name}'})")
                            else:
                                print(f"cpu test fail!({'{graph_name}'})")
                                print(f"cpu and tops comparision:")
                                if not isinstance(a[0], bool):
                                    print(f"a - b: {'{a - b}'}")
                                print(f"a: {'{a}'}")
                                print(f"b: {'{b}'}")
                        elif isinstance(a, list) and isinstance(b, list):
                            check_flag = True
                            if len(a) != len(b):
                                print(f"test check len: {'{len(a)}'}-{'{len(b)}'}")
                                raise ValueError("Error: len1 != len2")
                            for i in range(len(a)):
                                if isinstance(a[i], type(None)):
                                    continue
                                if not isinstance(a[i], torch.Tensor) or not isinstance(b[i], torch.Tensor):
                                    print(f"cpu test list type error({'{i}'})!({'{graph_name}'})")
                                    check_flag = False
                                else:
                                    if not torch.allclose(a[i], b[i], rtol=1e-03, equal_nan=True):
                                        print(f"cpu test fail({'{i}'})!({'{graph_name}'})")
                                        print(f"cpu and tops comparision{'{i}'}:")
                                        if a[i].type() != "torch.BoolTensor" and b[i].type() != "torch.BoolTensor":
                                            print(f"a[i] - b[i]: {'{a[i] - b[i]}'}")
                                        print(f"a[i]: {'{' +'a[i]' + '}'}")
                                        print(f"b[i]: {'{' +'b[i]' + '}'}")
                                        check_flag = False
                                    else:
                                        print(f"cpu test success({'{i}'})!({'{graph_name}'})")
                            if check_flag:
                                print(f"cpu test success(all-{'{len(a)}'})!({'{graph_name}'})")
                        else:
                            print(f"cpu test type error!({'{graph_name}'})")
                            print(type(a), type(b))
                        return
                """
                , strip=True
            )
        compile_graph_code.writeline(f"source_code = '''")
        compile_graph_code.splice(self.get_kernel_header(), strip=True)
        compile_graph_code.writeline("")
        compile_graph_code.writeline(f'std::shared_ptr<builder::Builder> build_sample() {"{"}')
        with compile_graph_code.indent():
            compile_graph_code.splice(self.gen_build_graph_code())
        compile_graph_code.writeline('}')
        compile_graph_code.writeline("")

        compile_graph_code.splice(self.gen_compile_func_code())
        compile_graph_code.splice(self.gen_load_func_code())
        compile_graph_code.splice(self.gen_run_func_code())
        compile_graph_code.writeline(f"'''")

        compile_graph_code.splice(
            f"""
                compile_job = TopsCompileJob(source_code)
                async_compile = AsyncCompileKernel()
                kernel_cpp_0 = async_compile.compile_kernel(compile_job)
                async_compile.wait(globals())
                del async_compile
            """
            , strip=True
        )
        return compile_graph_code.getvalue()

    def gen_tensor(self, prefix, tensor):
        if dipu_flag:
            res =  f"{prefix}({tuple(tensor.shape)}, {tensor.stride()}, device='dipu:{self.device_id}', dtype={tensor.dtype})"
        else:
            res =  f"{prefix}({tuple(tensor.shape)}, {tensor.stride()}, device='{tensor.device.type}', dtype={tensor.dtype})"
        # makes a copy of the tensor for view ops 
        if tops_check_precision and prefix == "empty_strided":
            res += ".contiguous()"
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
        if args:
            call_body.writeline(f"{', '.join(args)}, = args")
        call_body.writeline(f"args.clear()")
        call_body.writeline("")

        bufs = []
        none_bufs = []
        for i in range(len(self.output_args)):
            bufs.append("buf" + str(i))
            if not isinstance(self.output_args[i], type(None)):
                otensor = self.output_args[i].meta['val']
                call_body.writeline(bufs[-1] + " = " + self.gen_empty_tensor(otensor))
            else:
                none_bufs.append("buf" + str(i))
                call_body.writeline(bufs[-1] + " = " + (f"empty_strided((), ())"))

        call_body.writeline("")
        if dipu_flag:
            call_body.writeline(f"dipu_stream = torch_dipu.current_stream({self.device_id}).dipu_stream")

        arg_ptrs = []
        for i in range(len(args)):
            arg_ptrs.append("c_void_p(" + args[i] + ".data_ptr())")

        buf_ptrs = []
        for i in range(len(bufs)):
            if bufs[i] not in none_bufs:
                buf_ptrs.append("c_void_p(" + bufs[i] + ".data_ptr())")

        call_str = 'kernel_cpp_0('
        if dipu_flag:
            call_str += 'c_void_p(dipu_stream), '
        else:
            call_str += 'c_void_p(), '
        call_str += ", ".join(arg_ptrs + buf_ptrs) + ")"
        call_body.writeline(call_str)

        if tops_check_precision:
            call_body.writeline("import sys")
            call_body.writeline(f"if '{self.folder}' not in sys.path:")
            with call_body.indent():
                call_body.writeline(f"sys.path.insert(0, '{self.folder}')")
            call_body.writeline(f"from {self.graph_key[:4]} import {self.graph_key} as graph_module")
            call_body.writeline(f"cpu_module = graph_module()")
            call_body.writeline(f"cpu_res = cpu_module({', '.join(map(lambda s: s + '.cpu()', args))})")
            call_body.writeline(f"check_res(list(cpu_res), list(({', '.join(map(lambda s: s + '.cpu()', bufs))},)), '{self.graph_key}')")

        for arg in args:
            call_body.writeline(f'del {arg}')
        call_body.writeline("")
        
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
        
        main_body.writeline("")
        for i in range(0, len(self.input_args)):
            itensor = self.input_args[i].meta['val']
            main_body.writeline('arg' + str(i) + ' = ' + self.gen_random_tensor(itensor))

        args = []
        for i in range(len(self.input_args)):
            args.append('arg' + str(i))
        main_body.writeline("")
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
    def gen_args(args_dict, args, kwargs):
        src_code = IndentedBuffer()
        def convert_arg(arg):
            if isinstance(arg, Node):
                return args_dict[arg.name]
            elif isinstance(arg, bool):
                return str(arg).lower()
            elif isinstance(arg, numbers.Number):
                return arg
            elif isinstance(arg, str):
                return arg
            elif isinstance(arg, torch.dtype):
                return type_set[arg]
            elif isinstance(arg, torch.device):
                return str(arg)
            raise ValueError(f"unknown arg type({arg})")


        args_str = []
        kwargs_str = {}
        #name = process_name(node.name, node.target)
        #op_var = node.name

        for arg in args:
            if isinstance(arg, type(None)):
                continue
            args_str.append(convert_arg(arg))
        for k, v in kwargs.items():
            v = convert_arg(v) 
            kwargs_str[k] = v

        return src_code, args_str, kwargs_str

        '''
                    for k, v in kwargs.items():
                    if name == "Add" and k == "alpha":
                        kwargs_flatten.append(v)
                args += tuple(kwargs_flatten)


                if True:
                    if isinstance(node.meta['val'], list) or isinstance(node.meta['val'], tuple):
                        val = node.meta['val'][0]
                    else:
                        val = node.meta['val']

                    data_type = '' if isinstance(val, type(None)) else val.dtype.__str__()
                    
                    # handle mixed args(float and bool).
                    # todo check
                    #if data_type == "torch.bool":
                    #    data_type = args[i -1].meta['val'].dtype.__str__() if i > 0 and isinstance(args[i -1], Node) else "torch.float32"
                    
                    src_code.writeline("")
                    src_code.writeline(f"builder::Type {op_var}_const_value_type{count}({'{' + '1' + '}'}, {type_set[data_type]});")
                    
                    if data_type == 'torch.int64':
                        src_code.writeline(f'int {op_var}_const_value{count} = static_cast<int64_t>({str(arg)});')
                        build_type = "builder::Type({1}, s64_type)"
                    else:
                        src_code.writeline(f'float {op_var}_const_value{count} = static_cast<float>({str(arg)});')
                        build_type = "builder::Type({1}, f32_type)"
                    
                    src_code.writeline(f"builder::Op {op_var}_const{count} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value{count}), {build_type});")
                    
                    if data_type != "torch.int64" and data_type != "torch.float32":
                        src_code.writeline(f"{op_var}_const{count} = builder::Convert({op_var}_const{count}, {op_var}_const_value_type{count});")
                    
                    args_str.append(f'{op_var}_const{count}')
                    count += 1
                shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                data_type = node.meta['val'].dtype.__str__()
                src_code.writeline("")
                src_code.writeline(f"std::vector<int64_t> {op_var}_shape{count}{shape};")
                src_code.writeline(f"builder::Type {op_var}_type{count} = builder::Type({op_var}_shape{count}, {type_set[data_type]});")
                src_code.writeline("")
                args_str.append(f"{op_var}_type{count}")
                count += 1
            elif isinstance(arg, torch.fx.immutable_collections.immutable_list):
                pass
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
        '''


        # ---- not refact yet
        '''
        if name == "Reshape" and "complex" in args[0].name:
            args_str.append(node)
            src_code.writeline(f"builder::Op {args_dict[args[0].name]}_0 = builder::GetTupleElement({args_dict[args[0].name]}, 0);")
            src_code.writeline(f"builder::Op {args_dict[args[0].name]}_1 = builder::GetTupleElement({args_dict[args[0].name]}, 1);")
            args_str.append(f"{args_dict[args[0].name]}_0")
            args_str.append(f"{args_dict[args[0].name]}_1")
            args_str.append(str(args[1]).replace('[', '{').replace(']', '}'))
            return src_code, args_str
        else:
            args_str.append(node) if name in need_node else args_str
            gen_const_flag = False if name in not_gen_const else gen_const_flag
            if name in need_dict:
                args_str.append(args_dict)
                return src_code, args_str

        # TODO need a gen_kwargs function
        kwargs_flatten = []
        for k, v in kwargs.items():
            if name == "Add" and k == "alpha":
                kwargs_flatten.append(v)
        args += tuple(kwargs_flatten)

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
                src_code.writeline("")
                src_code.writeline(f"std::vector<int64_t> {op_var}_shape{count}{shape};")
                src_code.writeline(f"builder::Type {op_var}_type{count} = builder::Type({op_var}_shape{count}, {type_set[data_type]});")
                src_code.writeline("")
                args_str.append(f"{op_var}_type{count}")
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
            elif isinstance(args[i], str):
                args_str.append(args[i])
            else:
                if "squeeze" in node.name:
                    src_code.writeline("")
                    src_code.writeline(f"builder::Type {op_var}_axes_type{count}({'{' + '1' + '}'}, s64_type);")
                    
                    if "unsqueeze" in node.name:
                        src_code.writeline(f"std::vector<int64_t> {op_var}_axes_data{count} = {'{' + str(args[i]).split('[')[-1].split(']')[0] + '}'};")
                    
                    else:
                        src_code.writeline(f"std::vector<int64_t> {op_var}_axes_data{count} = {'{' + str(args[i]) + '}'};")

                    src_code.writeline(f"builder::Op {op_var}_axes{count} = builder::Const(hlir_builder, ({op_var}_axes_data{count}.data()), {op_var}_axes_type{count});")
                    src_code.writeline("")
                    
                    args_str.append(f'{op_var}_axes{count}')
                    
                    shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                    data_type = node.meta['val'].dtype.__str__()
                    src_code.writeline(f"builder::Type {op_var}_output_type{count}({shape}, {type_set[data_type]});")
                    src_code.writeline("")
                    
                    args_str.append(f"{op_var}_output_type{count}")
                    count += 1
                elif gen_const_flag:
                    if isinstance(node.meta['val'], list) or isinstance(node.meta['val'], tuple):
                        val = node.meta['val'][0]
                    else:
                        val = node.meta['val']

                    data_type = '' if isinstance(val, type(None)) else val.dtype.__str__()
                    
                    # handle mixed args(float and bool).
                    if data_type == "torch.bool":
                        data_type = args[i -1].meta['val'].dtype.__str__() if i > 0 and isinstance(args[i -1], Node) else "torch.float32"
                    
                    src_code.writeline("")
                    src_code.writeline(f"builder::Type {op_var}_const_value_type{count}({'{' + '1' + '}'}, {type_set[data_type]});")
                    
                    if data_type == 'torch.int64':
                        src_code.writeline(f'int {op_var}_const_value{count} = static_cast<int64_t>({str(args[i])});')
                        build_type = "builder::Type({1}, s64_type)"
                    else:
                        src_code.writeline(f'float {op_var}_const_value{count} = static_cast<float>({str(args[i])});')
                        build_type = "builder::Type({1}, f32_type)"
                    
                    src_code.writeline(f"builder::Op {op_var}_const{count} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value{count}), {build_type});")
                    
                    if data_type != "torch.int64" and data_type != "torch.float32":
                        src_code.writeline(f"{op_var}_const{count} = builder::Convert({op_var}_const{count}, {op_var}_const_value_type{count});")
                    
                    args_str.append(f'{op_var}_const{count}')
                    count += 1
                elif isinstance(args[i], int) or isinstance(args[i], float):
                    args_str.append(str(args[i]))
                else:
                    raise ValueError(f"Unsupported args type: {args[i]}!")
        return src_code, args_str
        '''

    @staticmethod
    def Clone(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = {x};"

    @staticmethod
    def Copy(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = {y};"
    
    @staticmethod
    def LiftFreshCopy(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = {x};"

    @staticmethod
    def Abs(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Abs({x});"

    @staticmethod
    def make_const_if_scalar(op_var, v, t=torch.float32, count=0):
        src_code = ""
        if isinstance(v, numbers.Number):
            src_code = f"{cxx_type_set[t]} {op_var}_const_value{count} = static_cast<{cxx_type_set[t]}>({v});\n"
            v = f"{op_var}_const{count}"
            src_code += f"builder::Type {op_var}_const_type{count}({{1}}, {type_set[t]});\n"
            src_code += f"builder::Op {v} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value{count}), {op_var}_const_type{count});\n"
        return src_code, v
    
    @staticmethod
    def make_type(op_var, t, shape=[1], count=0):
        t = type_set[t] if isinstance(t, torch.dtype) else t
        shape = f"{{{str(shape).split('[')[-1].split(']')[0]}}}"
        src_code = f"std::vector<int64_t> {op_var}_shape{count}{shape};\n"
        src_code += f"builder::Type {op_var}_type{count}({shape}, {t});\n"
        return src_code, f"{op_var}_type{count}"

    @staticmethod
    # TODO mul + add scaled_y should handle in conversion
    def Add(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code += f"builder::Op {op_var} = builder::Add({x}, {y});"
        return src_code
    
    @staticmethod
    def Convert(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_type(op_var, y, shape)     
        src_code += f"builder::Op {op_var} = builder::Convert({x}, {y});"
        return src_code
    
    @staticmethod
    def Div(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code += f"builder::Op {op_var} = builder::Div({x}, {y});"
        return src_code
    
    @staticmethod
    def Sub(op_var, shape, dtype, x,  y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code += f"builder::Op {op_var} = builder::Sub({x}, {y});"
        return src_code

    @staticmethod
    def Mul(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code += f"builder::Op {op_var} = builder::Mul({x}, {y});"
        return src_code
    
    @staticmethod
    def Bmm(op_var, node, args_dict):
        args = node.args
        args_str = []
        src_code = ""

        for i in range(0, len(args)):
            tmp_data_type = args[i].meta['val'].dtype.__str__()
            if tmp_data_type != 'torch.float32':
                tmp_shape = '{' + str(args[i].meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                src_code += f"builder::Type {args_dict[args[i].name]}_dot_type({tmp_shape}, f32_type);\n"
                src_code += f"builder::Op {args_dict[args[i].name]}_tmp = builder::Convert({args_dict[args[i].name]}, {args_dict[args[i].name]}_dot_type);\n"
                args_str.append(f"{args_dict[args[i].name]}_tmp")
            else:
                args_str.append(f"{args_dict[args[i].name]}")

        src_code += f"builder::DotDimensionNumbers {op_var}_dims_attr({'{0}'}, {'{0}'}, {'{2}'}, {'{1}'});\n"
        src_code += f"builder::Op {op_var}_tmp = builder::DotGeneral({', '.join(args_str)}, {op_var}_dims_attr);\n"

        data_type = node.meta['val'].dtype.__str__()            
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code += f"builder::Type {op_var}_type({shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var} = builder::Convert({op_var}_tmp, {op_var}_type);"

        return src_code

    @staticmethod
    def DotGeneral(op_var, lhs, rhs, dot_dimension_numbers_str):
        src_code = ""
        src_code += f"builder::DotDimensionNumbers {op_var}_dims_attr({dot_dimension_numbers_str});\n"
        src_code += f"builder::Op {op_var} = builder::DotGeneral({lhs}, {rhs}, {op_var}_dims_attr);\n"
        src_code += f"""{op_var}.SetAttribute("op_type", builder::Attribute("DotInference"));"""
        return src_code

    @staticmethod
    def Dot(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::Dot({x}, {y});"

    @staticmethod
    def Gemm(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::Gemm({{{x}, {y}}});"
    
    @staticmethod
    def Max(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::Max({x}, {y});"
    
    @staticmethod
    def Less(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::Less({x}, {y});"
    
    @staticmethod
    def Equal(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y)
        src_code += f"builder::Op {op_var} = builder::Equal({x}, {y});"
        return src_code
        
    @staticmethod
    def LessEqual(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y)
        src_code += f"builder::Op {op_var} = builder::LessEqual({x}, {y});"
        return src_code
    
    @staticmethod
    def NotEqual(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y)
        src_code += f"builder::Op {op_var} = builder::NotEqual({x}, {y});"
        return src_code
    
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
        return f"builder::Op {op_var} = builder::Sigmoid({x});"
    
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

        if data_type == "torch.int64":
            src_code += f'int {op_var}_const_value = static_cast<int64_t>({node.args[0]});'
            src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value), {op_var}_scalar_type);\n"
        else:
            src_code += f"float {op_var}_const_value = static_cast<float>({node.args[0]});\n"
            if data_type == "torch.float32":
                src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value), {op_var}_scalar_type);\n"
            else:
                src_code += f"builder::Op {op_var}_t = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value), builder::Type({'{' + '1' +'}'}, builder::PrimitiveType::F32()));\n"
                src_code += f"builder::Op {op_var} = builder::Convert({op_var}_t, {op_var}_scalar_type);\n"

        return src_code
    
    @staticmethod
    def Getitem(op_var, node, *args_str):
        src_code = f"builder::Op {op_var} = builder::GetTupleElement({args_str[0]}, {int(node.args[1])});"
        return src_code
    
    @staticmethod
    def NativeDropout(op_var, node, *args):
        src_code = f"builder::Op {op_var}_dropout = builder::Dropout({', '.join(args)});\n"
        data_type = node.meta['val'][0].dtype.__str__()            
        shape = '{' + str(node.meta['val'][0].shape).split('[')[-1].split(']')[0] + '}'
        src_code += f"std::vector<int64_t> {op_var}_const_shape{shape};\n"
        src_code += f"builder::Type {op_var}_const_type({op_var}_const_shape, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var}_const = builder::Const(hlir_builder, 0, {op_var}_const_type);\n"
        src_code += f"builder::Op {op_var}_notequal = builder::NotEqual({op_var}_dropout, {op_var}_const);\n"
        src_code += f"std::vector<builder::Op> {op_var}_outputs = {'{' + op_var + '_dropout, ' + op_var + '_notequal' + '}'};\n"
        src_code += f"builder::Op {op_var} = builder::Tuple({op_var}_outputs);\n"
        return src_code

    @staticmethod
    def Where(op_var, *args):
        return f"builder::Op {op_var} = builder::Select({', '.join(args)});"

    @staticmethod
    def ZerosLike(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()            
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"builder::Op {op_var} = builder::ZerosLike({args_str[0]}, {type_set[data_type]}, {shape});"
        return src_code

    @staticmethod
    def EmptyLike(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__() 
        src_code = f"builder::Op {op_var} = builder::EmptyLike({args_str[0]}, {type_set[data_type]}, {{0}});"
        return src_code
    
    # temporary fake implementation.
    @staticmethod
    def Bernoulli(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"builder::Op {op_var} = builder::OnesLike({args_str[0]}, {type_set[data_type]}, {shape});"
        return src_code

    @staticmethod
    def NewEmptyStrided(op_var, *args_str):
        return f"builder::Op {op_var} = builder::EmptyLike({args_str[0]});"

    @staticmethod
    def OnesLike(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"builder::Op {op_var} = builder::OnesLike({args_str[0]}, {type_set[data_type]}, {shape});"
        return src_code
    
    @staticmethod
    def Full(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        src_code = f"std::vector<int64_t> {op_var}_in_shape{args_str[0]};\n"
        src_code += f"builder::Type {op_var}_type({op_var}_in_shape, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, {node.args[1]}, {op_var}_type);"
        return src_code
    
    @staticmethod
    def FullLike(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"  builder::Op {op_var} = builder::FullLike({args_str[0]}, {str(node.args[1])}, {type_set[data_type]}, {shape});"
        return src_code
    
    @staticmethod
    def Transpose(op_var, *args):
        if len(args) == 1:
            list(args).append('{1, 0}')
        return f"builder::Op {op_var} = builder::Transpose({', '.join(args)});"
    
    @staticmethod
    def Hardswish(op_var, x):
        return f"builder::Op {op_var} = builder::HardSwish({x});"

    @staticmethod
    def Hardswish_Grad(op_var, x, y):
        a, b, c = 3.0, 6.0, 6.0
        return f"builder::Op {op_var} = builder::HardSwishGrad({x}, {y}, {a}, {b}, {c});"

    @staticmethod
    def Reshape(op_var, node, *args_str):
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        src_code = f"builder::Type {op_var}_reshape_shape({shape}, {type_set[data_type]});\n"
        tmp = f'{op_var}_reshape_shape'
        if len(args_str) == 2:
            src_code += f"builder::Op {op_var} = builder::Reshape({args_str[0]}, {tmp});"
        else:
            src_code += f"builder::Op {op_var}_0 = builder::Reshape({args_str[0]}, {tmp});\n"
            src_code += f"builder::Op {op_var}_1 = builder::Reshape({args_str[1]}, {tmp});\n"
            src_code += f"std::vector<builder::Op> {op_var}_outputs = {'{' + op_var +'_0, ' + op_var + '_1' + '}'};\n"
            src_code += f"builder::Op {op_var} = builder::Tuple({op_var}_outputs);"
        return src_code
    
    @staticmethod
    def Expand(op_var, node, *args_str):
        dims = []
        for i in range(0, len(node.meta['val'].shape)):
            dims.append(str(i))
        broadcast_dims = '{' + ', '.join(dims) + '}'
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        src_code = f"builder::Type {op_var}_expand_type({shape}, {type_set[data_type]});\n"
        src_code += f"auto {op_var} = BroadcastInDim({args_str[0]}, {broadcast_dims}, {op_var}_expand_type);"
        return src_code
    
    @staticmethod
    def Squeeze(op_var, *args):
        return f"builder::Op {op_var} = builder::Squeeze({', '.join(args)});"
    
    @staticmethod
    def Unsqueeze(op_var, *args):
        return f"builder::Op {op_var} = builder::Unsqueeze({', '.join(args)});"

    @staticmethod
    def ReduceMean(op_var, *args):
        src_code = ''
        if len(args) == 3:
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args[0]}, {args[2]}, {args[1]});"
        elif len(args) == 2:
            keepdim = 'false'
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args[0]}, {keepdim}, {args[1]});"
        elif len(args) == 1:
            src_code = f"builder::Op {op_var} = builder::ReduceMean({args[0]});"
        else:   
            ValueError("Reducemean args num error!")
        return src_code

    @staticmethod
    def ReduceMax(op_var, *args):
        src_code = ''
        if len(args) == 3:
            src_code = f"builder::Op {op_var} = builder::ReduceMax({args[0]}, {args[2]}, {args[1]});"
        elif len(args) == 2:
            keepdim = 'false'
            src_code = f"builder::Op {op_var} = builder::ReduceMax({args[0]}, {keepdim}, {args[1]});"
        elif len(args) == 1:
            src_code = f"builder::Op {op_var} = builder::ReduceMax({args[0]});"
        else:
            ValueError("ReduceMax args num error!")
        return src_code

    @staticmethod
    def ReduceSum(op_var, node, *args):
        input_shape = '{' + ', '.join(map(str, node.args[0].meta['val'].shape)) + '}'
        input_type = node.args[0].meta['val'].dtype.__str__()
        output_type =  node.meta['val'].dtype.__str__()

        src_code = ''
        if input_type != output_type:
            src_code += f"{args[0]} = builder::Convert({args[0]}, builder::Type({input_shape}, {type_set[output_type]}));"

        if len(args) == 3:
            src_code += f"builder::Op {op_var} = builder::ReduceSum({args[0]}, {args[2]}, {args[1]});"
        elif len(args) == 2:
            keepdim = 'false'
            src_code += f"builder::Op {op_var} = builder::ReduceSum({args[0]}, {keepdim}, {args[1]});"
        elif len(args) == 1:
            src_code += f"builder::Op {op_var} = builder::ReduceSum({args[0]});"
        else:
            ValueError("ReduceSum args num error!")

        return src_code

    @staticmethod
    def Scatter(op_var, node, *args_str):
        new_args_str = []
        new_args_str.append(args_str[0])
        new_args_str.append(args_str[1])
        new_args_str.append(args_str[2])
        
        if isinstance(node.args[3], float):
            src_code = f"const float {op_var}_value = {str(node.args[3])};\n"
        else:
            src_code = f"const int {op_var}_value = {str(node.args[3])};\n"
        
        new_args_str.append(f'{op_var}_value')

        src_code += f"auto {op_var} = enflame::Scatter(hlir_builder, {', '.join(new_args_str)});"

        return src_code
    
    @staticmethod
    def Gather(op_var, node, *args_str):
        data_type = node.meta['val'].dtype.__str__()
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        new_args_str = []
        new_args_str.append(args_str[0])
        new_args_str.append(args_str[2])
        new_args_str.append(args_str[1])
        
        src_code = f"builder::Type {op_var}_gather_type({shape}, {type_set[data_type]});\n"
        new_args_str.append(f"{op_var}_gather_type")
        
        src_code += f"auto {op_var} = enflame::Gather(hlir_builder, {', '.join(new_args_str)});"

        return src_code
    
    @staticmethod
    def Slice(op_var, node, args_dict):
        args = node.args
        in_shape = '{' + ', '.join(map(str, args[0].meta['val'].shape)) + '}'
        out_shape = '{' + ', '.join(map(str, node.meta['val'].shape)) + '}'
        
        shape = args[0].meta['val'].shape
        rank = len(shape)
        dim = int(args[1])
        
        src_code = "\n"
        if in_shape != out_shape:
            start_indice = (int(args[2]) + shape[dim]) % shape[dim]
            limit_indice = (int(args[3]) + shape[dim]) % shape[dim] if int(args[3]) < shape[dim] else shape[dim]
            
            src_code += f"auto {op_var} = builder::SliceInDim({args_dict[args[0].name]}, {start_indice}, {limit_indice}, {1}, {dim});"
            
        else:
            start_indices = [0 for x in range(0, rank)]    
            start_indices = '{' + ', '.join(map(str, start_indices)) + '}'   
            limit_indices = in_shape
            stride = [1 for x in range(0, rank)]
            stride = '{' + ', '.join(map(str, stride)) + '}'   
            src_code += f"auto {op_var} = builder::Slice({args_dict[args[0].name]}, {start_indices}, {limit_indices}, {stride});"
        
        return src_code

    @staticmethod
    def SliceScatter(op_var, node, *args_str):
        operand_shape = node.args[0].meta['val'].shape
        dim = node.args[2] if len(node.args) > 2 else 0
        start = node.args[3] if len(node.args) > 3 else 0
        end = (node.args[4] + operand_shape[dim]) % operand_shape[dim] if len(node.args) > 4 and node.args[4] < operand_shape[dim] else operand_shape[dim]
        step = node.args[5] if len(node.args) > 5 else 1
        
        if end != operand_shape[dim]:
            print(f"Warning: EnflameOverrides.SliceScatter encounter unsupported end value: {end},  this will affect precision!")
            
        if step != 1:
            print(f"Warning: EnflameOverrides.SliceScatter encounter unsupported step value: {step}, this will affect precision!")
        
        const_shape = "{1}"
        const_value = "{0}"
        src_code = f"builder::Type {op_var}_index_type({const_shape}, builder::PrimitiveType::S64());\n"
        src_code += f"int64_t {op_var}_start_index_data = {const_value};\n"
        src_code += f"builder::Op {op_var}_start_index = builder::Const(hlir_builder, &{op_var}_start_index_data, {op_var}_index_type);\n"
        src_code += f"int64_t {op_var}_start_index_data_{dim} = {start};\n"
        src_code += f"builder::Op {op_var}_start_index_{dim} = builder::Const(hlir_builder, &{op_var}_start_index_data_{dim}, {op_var}_index_type);\n"
        
        start_indices = [f"{op_var}_start_index" for i in range(len(operand_shape))]
        start_indices[dim] = f"{op_var}_start_index_{dim}"
        
        src_code += f"builder::Op {op_var} = builder::DynamicUpdateSlice({args_str[0]}, {args_str[1]}, {{{', '.join(start_indices)}}});"
        
        return src_code

    @staticmethod
    def Index(op_var, node, *args):
        src_code = ""
        indices_list = args[-1].strip("{ }").split(", ")   
        if len(indices_list) == 1:
            indices = indices_list[0]
        else:
            src_code += f"builder::Op {op_var}_indices = builder::Concatenate({{{', '.join(indices_list)}}}, 0);\n"
            indices = f"{op_var}_indices"
        src_code += f"std::vector<int64_t> {op_var}_offset_dims = {{{indices}.GetType().GetRank()}};\n" \
                    f"std::vector<int64_t> {op_var}_collapsed_slice_dims = {{0}};\n" \
                    f"std::vector<int64_t> {op_var}_start_index_map = {{0}};\n" \
                    f"int64_t {op_var}_index_vector_dim = {indices}.GetType().GetRank();\n" \
                    f"auto {op_var}_dimension_numbers = builder::GatherDimensionNumbers({op_var}_offset_dims, {op_var}_collapsed_slice_dims, {op_var}_start_index_map, {op_var}_index_vector_dim);\n" \
                    f"std::vector<int64_t> {op_var}_slice_sizes = {args[0]}.GetType().GetShape();\n" \
                    f"{op_var}_slice_sizes[0] = 1;\n" \
                    f"builder::Op {op_var} = builder::Gather({args[0]}, {indices}, {op_var}_dimension_numbers, {op_var}_slice_sizes);\n"
        return src_code

    @staticmethod
    def Select(op_var, node, args_dict):
        args = node.args
        shape = args[0].meta['val'].shape
        rank = len(shape)
        dim = int(args[1])
        
        index = (int(args[2]) + shape[dim]) % shape[dim]
        
        start_indices = [0 for i in range(0, rank)]  
          
        start_indices[dim] = index
        start_indices = '{' + ', '.join(map(str, start_indices)) + '}'   
        
        limit_indices = [x for x in shape]
        limit_indices[dim] = index + 1
        limit_indices = '{' + ', '.join(map(str, limit_indices)) + '}'
        
        stride = [1 for x in range(0, rank)]
        stride = '{' + ', '.join(map(str, stride)) + '}' 
        
        src_code = f"auto {op_var}_t = builder::Slice({args_dict[args[0].name]}, {start_indices}, {limit_indices}, {stride});\n"

        src_code += f"builder::Type {op_var}_axes_type({'{1}'}, s64_type);\n"
        src_code += f"std::vector<int64_t> {op_var}_axes_data = {'{' + str(dim) + '}'};\n"
        src_code += f"builder::Op {op_var}_axes = builder::Const(hlir_builder, ({op_var}_axes_data.data()), {op_var}_axes_type);\n"
        src_code += f"auto {op_var} = builder::Squeeze({op_var}_t, {op_var}_axes);"
        
        return src_code
    
    @staticmethod
    def Batch_Norm(op_var, node, *args_str):
        args_str_tmp = args_str[:5]
        src_code = f"auto {op_var} = enflame::BatchNorm(hlir_builder, {', '.join(args_str_tmp)}, 1, true, {str(node.args[6])}, {str(node.args[7])});"     
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
        src_code += f'builder::Op {op_var} = builder::Conv2D({op_var}_inputs, {group}, "NOTSET", "NCHW", {stride}, {padding}, {dilation});'

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
        
        src_code = f"auto {op_var} = enflame::Conv2D_Grad(hlir_builder, {','.join(new_args_str)}, {bias}, {stride}, {padding}, {dilation});"

        return src_code

    @staticmethod
    def MaxPool2D(op_var, node, *args_str):
        args = node.args
        
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
                       
        src_code = f"builder::Op {op_var} = enflame::MaxPool2D(hlir_builder, {args_str[0]}, {ksize}, {stride}, {padding}, {shape});"

        return src_code
    
    @staticmethod
    def MaxPool2D_Grad(op_var, node, *args_str):
        args = node.args    
        
        ksize = str(args[2]).replace('[','{').replace(']','}')
        strides = str(args[3]).replace('[','{').replace(']','}')
        padding = str(args[4]).replace('[','{').replace(']','}')
        
        src_code = f"auto {op_var} = enflame::MaxPool2D_Grad(hlir_builder, {args_str[0]}, {args_str[1]}, {ksize}, {strides}, {padding});"
        
        return src_code
    

    # redispatch
    @staticmethod
    def AvgPool2D(op_var, *args_str):
        if len(args_str) != 2 or args_str[1] != "{1, 1}":
            raise ValueError(f"Op AvgPool2D error: {args_str}.")
        axis = "{2, 3}"
        src_code = f"builder::Op {op_var} = builder::ReduceMean({args_str[0]}, true, {axis});"
        
        return src_code
    
    # redispatch
    @staticmethod
    def AvgPool2D_Grad(op_var, node, *args_str):
        dims = []
        for i in range(0, len(node.meta['val'].shape)):
            dims.append(str(i))
        broadcast_dims = '{' + ', '.join(dims) + '}'
        
        shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        
        value = node.meta['val'].shape[2] * node.meta['val'].shape[3]
                
        src_code = f"builder::Type {op_var}_expand_type({shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var}_tmp = BroadcastInDim({args_str[0]}, {broadcast_dims}, {op_var}_expand_type);\n"
        
        src_code += f"float {op_var}_const_value = static_cast<float>({str(value)});\n"
        src_code += f"builder::Op {op_var}_const = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value), builder::Type({'{' + '1' +'}'}, f32_type));\n"
        
        src_code += f"builder::Op {op_var} = builder::Div({op_var}_tmp, {op_var}_const);\n"
        
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
                   f");"

        return src_code
    
    # [a + bi] ===> tops.tuple(a, bi)
    @staticmethod
    def Complex(op_var, node, args_dict):
        args = node.args
        shape = '{' + str(args[0].meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        
        src_code = f"std::vector<int64_t> {op_var}_in_shape{shape};\n"
        
        src_code += f"int {op_var}_in_shape_size = {op_var}_in_shape.size();\n\n"
        
        src_code += f"std::vector<int64_t> {op_var}_part0_start_indices({op_var}_in_shape_size, 0);\n"
        src_code += f"auto {op_var}_part0_limit_indices =  {op_var}_in_shape;\n"
        src_code += f"{op_var}_part0_limit_indices[{op_var}_in_shape_size - 1]--;\n"
        
        src_code += f"std::vector<int64_t> {op_var}_part1_start_indices({op_var}_in_shape_size, 0);\n"
        src_code += f"{op_var}_part1_start_indices[{op_var}_in_shape_size - 1] = 1;\n"
        
        src_code += f"std::vector<int64_t> {op_var}_stride({op_var}_in_shape_size, 1);\n\n"

        src_code += f"builder::Op {op_var}_split0 = builder::Slice({args_dict[args[0].name]}, {op_var}_part0_start_indices, {op_var}_part0_limit_indices, {op_var}_stride);\n"
        src_code += f"builder::Op {op_var}_split1 = builder::Slice({args_dict[args[0].name]}, {op_var}_part1_start_indices, {op_var}_in_shape, {op_var}_stride);\n"
        
        out_shape = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
        data_type = args[0].meta['val'].dtype.__str__()
        
        src_code += f"builder::Type {op_var}_reshape_type({out_shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var}_tmp0 = builder::Reshape({op_var}_split0, {op_var}_reshape_type);\n"
        src_code += f"builder::Op {op_var}_tmp1 = builder::Reshape({op_var}_split1, {op_var}_reshape_type);\n"

        t = '{'
        t += f"{op_var}_tmp0, {op_var}_tmp1"
        t += '}'
        
        src_code += f"std::vector<builder::Op> {op_var}_outputs{t};\n"

        src_code += f"builder::Op {op_var} = builder::Tuple({op_var}_outputs);"
        
        return src_code

    # tops.tuple(a, bi)====>[a,b]
    @staticmethod
    def Viewasreal(op_var, node, x):
        src_code = f"builder::Op {op_var}_real = builder::GetTupleElement({x}, 0);\n"
        src_code += f"builder::Op {op_var}_imag = builder::GetTupleElement({x}, 1);\n"
        
        out_shape = '{' + str(list(node.meta['val'].shape)[:-1] + [1]).split('[')[-1].split(']')[0] + '}'
        data_type = node.meta['val'].dtype.__str__()
        
        src_code += f"builder::Type {op_var}_reshape_type({out_shape}, {type_set[data_type]});\n"
        src_code += f"builder::Op {op_var}_tmp0 = builder::Reshape({op_var}_real, {op_var}_reshape_type);\n"
        src_code += f"builder::Op {op_var}_tmp1 = builder::Reshape({op_var}_imag, {op_var}_reshape_type);\n"
        
        t = '{'
        t += f"{op_var}_tmp0, {op_var}_tmp1"
        t += '}'

        src_code += f"std::vector<builder::Op> {op_var}_real_imag = {t};\n"
        dimension = len(node.meta['val'].shape)-1
        src_code += f'builder::Op {op_var} = builder::Concatenate({op_var}_real_imag, {dimension});'

        return src_code

    #(a + bi)(c + di) = (ac -bd) + (ad + bd)i
    @staticmethod
    def Complexmul(op_var, node, x, y):
        src_code = f"builder::Op {op_var}_xreal = builder::GetTupleElement({x}, 0);\n"
        src_code += f"builder::Op {op_var}_ximag = builder::GetTupleElement({x}, 1);\n"

        src_code += f"builder::Op {op_var}_yreal = builder::GetTupleElement({y}, 0);\n"
        src_code += f"builder::Op {op_var}_yimag = builder::GetTupleElement({y}, 1);\n"

        src_code += f"builder::Op {op_var}_xreal_yreal = builder::Mul({op_var}_xreal, {op_var}_yreal);\n"
        src_code += f"builder::Op {op_var}_ximag_yimag = builder::Mul({op_var}_ximag, {op_var}_yimag);\n"

        src_code += f"builder::Op {op_var}_xreal_yimag = builder::Mul({op_var}_xreal, {op_var}_yimag);\n"
        src_code += f"builder::Op {op_var}_ximag_yreal = builder::Mul({op_var}_ximag, {op_var}_yreal);\n"

        src_code += f"builder::Op {op_var}_mul_real = builder::Sub({op_var}_xreal_yreal, {op_var}_ximag_yimag);\n"
        src_code += f"builder::Op {op_var}_mul_imag = builder::Add({op_var}_xreal_yimag, {op_var}_ximag_yreal);\n"

        t = '{'
        t += f"{op_var}_mul_real, {op_var}_mul_imag"
        t += '}'

        src_code += f"std::vector<builder::Op> {op_var}_outputs {t};\n"
        src_code += f"builder::Op {op_var} = builder::Tuple( {op_var}_outputs);"
        
        return src_code

    @staticmethod
    def Concatenate(op_var, node, args_dict):
        args = node.args
        args_str = []

        for arg in args[0]:
            arg_shape = arg.meta['val'].shape
            if 0 in arg_shape:
                continue
            args_str.append(args_dict[arg.name])
        
        dim = (node.args[1] + len(node.meta["val"].shape)) % len(node.meta["val"].shape)
            
        return f"builder::Op {op_var} = builder::Concatenate({'{' + ', '.join(args_str) + '}'}, {dim});"

    @staticmethod
    def Softmax(op_var, x, y, z):
        return f"builder::Op {op_var} = builder::Softmax({x}, {y}, {z});"

    @staticmethod
    def Logsoftmax(op_var, x, y, z):
        return f"builder::Op {op_var} = builder::Softmax({x}, {y}, {z}, true);"

    @staticmethod
    def Gelu(op_var, shape, dtype, x, approximate, **kwargs):
        return f"builder::Op {op_var} = builder::Gelu({x}, {approximate});"

    @staticmethod
    def Gelu_Grad(op_var, shape, dtype, x, y, approximate):
        return f"builder::Op {op_var} = builder::GeluGrad({x}, {y}, {approximate});"

    @staticmethod
    def Iota(op_var, node_shape, node_dtype, length, **kwargs_list):
        src_code, var_type = EnflameOverrides.make_type(op_var, node_dtype, node_shape)
        src_code += f"builder::Op {op_var} = builder::Iota(hlir_builder, 0, {var_type});\n"
        return src_code
