import os
import numbers
from torch._inductor.utils import IndentedBuffer

import torch

from typing import Any
from torch.fx.node import Node

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
                torch.float16: "float_t",
                torch.float64: "double_t",
                torch.double: "double_t",
                torch.int32: "int32_t",
                torch.int: "int32_t",
                torch.int64: "int64_t",
                torch.long: "int64_t",
                torch.bool: "bool"}


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
        self.input_args = []
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
        self.build_graph_code.writeline(
            f"std::vector<int64_t> {self.args_dict[name]}_in_shape{in_shape};")
        self.build_graph_code.writeline(
            f"builder::Type {self.args_dict[name]}_input_type({self.args_dict[name]}_in_shape, {type_set[data_type]});")
        self.build_graph_code.writeline(
            f"builder::Op {self.args_dict[name]} = hlir_builder->CreateInput({self.args_dict[name]}_input_type);")
        self.build_graph_code.writeline("")

    def get_attr(self, name, target, args, kwargs):
        assert isinstance(target, str)
        if name not in self.args_dict.keys():
            op_var = self.args_dict[name] = name + \
                '_' + str(len(self.args_dict))
        attr = self.fetch_attr(target)
        assert (isinstance(attr, torch.Tensor))
        if attr.size():
            data = str(attr)[str(attr).find(
                '(') + 1: -1].replace('[', '{').replace(']', '}')
            shape = '{' + str(attr.shape).split('[')[-1].split(']')[0] + '}'
            self.build_graph_code.writeline(
                f"builder::Type {op_var}_type({shape}, {type_set[attr.dtype]});")
            self.build_graph_code.writeline(
                f"std::vector<{cxx_type_set[attr.dtype]}> {op_var}_data{data};")
            self.build_graph_code.writeline(
                f"builder::Op {op_var} = builder::Const(hlir_builder, {op_var}_data.data(), {op_var}_type);")
        else:
            self.build_graph_code.writeline(
                f"builder::Type {op_var}_type({{1}}, {type_set[attr.dtype]});")
            self.build_graph_code.writeline(
                f"builder::Op {op_var} = builder::Const(hlir_builder, {attr}, {op_var}_type);")

    def call_function(self, name, target, args, kwargs):
        if name not in self.args_dict.keys():
            op_var = self.args_dict[name] = name + \
                '_' + str(len(self.args_dict))
        arg_code, args_list, kwargs_list = EnflameOverrides.gen_args(
            self.args_dict, args, kwargs)
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

        # some of the nodes have a list/tuple of faketensor in meta['val']
        try:
            node_shape = self.cur_node.meta['val'].shape
        except Exception:
            node_shape = None
        try:
            node_dtype = self.cur_node.meta['val'].dtype
        except Exception:
            node_dtype = None

        op_code = getattr(self.override, real_op)(
            op_var, node_shape, node_dtype, *args_list, **kwargs_list)

        self.build_graph_code.splice(arg_code)
        self.build_graph_code.splice(op_code)
        self.build_graph_code.writeline("")

        return

    def output(self, name, target, args, kwargs):
        self.inplace_dict = kwargs
        for i in range(0, len(args[0])):
            self.output_args.append(args[0][i])

    def run_node(self, n: Node) -> Any:
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
        shape = '{' + \
            str(self.cur_node.meta['val'].shape).split(
                '[')[-1].split(']')[0] + '}'
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
            """, strip=True
        )
        return self.import_code.getvalue()

    def gen_build_graph_code(self):
        graph_code = IndentedBuffer()
        graph_code.writelines(
            [
                'auto hlir_builder = std::make_shared<builder::Builder>();',
                'hlir_builder->SetShapeInference(true);',
                'auto f16_type = builder::PrimitiveType::F16();',
                'auto f32_type = builder::PrimitiveType::F32();',
                'auto s64_type = builder::PrimitiveType::S64();'
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
        graph_code.writeline(
            f'hlir_builder->SetOutput({"{" + ", ".join(output_str) + "}"});')
        graph_code.writeline("")
        graph_code.writeline('return hlir_builder;')
        return graph_code

    def gen_compile_func_code(self):
        compile_func_body = IndentedBuffer()
        with compile_func_body.indent():
            compile_func_body.splice(
                """
                    auto hlir_builder = build_sample();
                    compile(hlir_builder, &exe_ptr, compile_bin_path);
                """, strip=True
            )
        compile_func = IndentedBuffer()
        compile_func.writelines(
            [
                'topsExecutable_t exe_ptr;\n',
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
        func_body.writeline('std::vector<void *> input_ptrs;')
        for i in range(0, len(self.input_args)):
            func_body.writeline(f'input_ptrs.emplace_back(inputs_ptr[{str(i)}]);')

        func_body.writeline("")
        func_body.writeline("std::vector<void *> output_ptrs;")
        output_ptr_count = 0
        for i in range(0, len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                func_body.writeline(f'output_ptrs.emplace_back(outputs_ptr[{output_ptr_count}]);')
                output_ptr_count += 1
        func_body.writeline("")

        func_body.writeline(f'run(exe_ptr, dipu_stream, input_ptrs, output_ptrs, {self.device_id}, {"true" if dipu_flag else "false"});')

        run_func_code = IndentedBuffer()
        run_func_code.writeline(f'extern "C" void run(void *dipu_stream, void **inputs_ptr, void **outputs_ptr) {"{"}')
        with run_func_code.indent():
            run_func_code.splice(func_body)
        run_func_code.splice('}')
        return run_func_code

    def gen_load_func_code(self):
        func_body = IndentedBuffer()
        func_body.writeline("load(&exe_ptr, compile_bin_path);")

        run_func_code = IndentedBuffer()
        run_func_code.writeline(
            f'extern "C" void load(const wchar_t *compile_bin_path){"{"}')

        with run_func_code.indent():
            run_func_code.splice(func_body)
        run_func_code.splice('}')

        return run_func_code

    def get_kernel_header(self):
        return """
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
                """, strip=True
            )
        compile_graph_code.writeline("source_code = '''")
        compile_graph_code.splice(self.get_kernel_header(), strip=True)
        compile_graph_code.writeline("")
        compile_graph_code.writeline(
            f'std::shared_ptr<builder::Builder> build_sample() {"{"}')
        with compile_graph_code.indent():
            compile_graph_code.splice(self.gen_build_graph_code())
        compile_graph_code.writeline('}')
        compile_graph_code.writeline("")

        compile_graph_code.splice(self.gen_compile_func_code())
        compile_graph_code.splice(self.gen_load_func_code())
        compile_graph_code.splice(self.gen_run_func_code())
        compile_graph_code.writeline("'''")

        compile_graph_code.splice(
            """
                compile_job = TopsCompileJob(source_code)
                async_compile = AsyncCompileKernel()
                kernel_cpp_0 = async_compile.compile_kernel(compile_job)
                async_compile.wait(globals())
                del async_compile
            """, strip=True
        )
        return compile_graph_code.getvalue()

    def gen_tensor(self, prefix, tensor):
        if dipu_flag:
            res = f"{prefix}({tuple(tensor.shape)}, {tensor.stride()}, device='dipu:{self.device_id}', dtype={tensor.dtype})"
        else:
            res = f"{prefix}({tuple(tensor.shape)}, {tensor.stride()}, device='{tensor.device.type}', dtype={tensor.dtype})"
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
            args.append(self.input_args[i].name)
        if args:
            call_body.writeline(f"{', '.join(args)}, = args")
        call_body.writeline("args.clear()")
        call_body.writeline("")

        bufs = []
        none_bufs = []
        for i in range(len(self.output_args)):
            if not isinstance(self.output_args[i], type(None)):
                bufs.append(self.output_args[i].name)
                if self.output_args[i] not in self.input_args and bufs[-1] not in self.inplace_dict.keys():
                    otensor = self.output_args[i].meta['val']
                    call_body.writeline(bufs[-1] + " = " + self.gen_empty_tensor(otensor))
            else:
                bufs.append("buf" + str(i))
                none_bufs.append(bufs[-1])
                call_body.writeline(
                    bufs[-1] + " = " + ("empty_strided((), ())"))
        for i in range(len(bufs) - len(self.inplace_dict), len(bufs)):
            bufs[i] = self.inplace_dict[bufs[i]]

        call_body.writeline("")

        arg_ptrs = []
        for i in range(len(args)):
            arg_ptrs.append("c_void_p(" + args[i] + ".data_ptr())")

        buf_ptrs = []
        for i in range(len(bufs)):
            if bufs[i] not in none_bufs:
                buf_ptrs.append("c_void_p(" + bufs[i] + ".data_ptr())")

        call_body.writeline(f"args_ptr_type = c_void_p * {len(arg_ptrs)}")
        call_body.writeline(f"bufs_ptr_type = c_void_p * {len(buf_ptrs)}")
        call_body.writeline(f"args_ptr = args_ptr_type({','.join(arg_ptrs)})")
        call_body.writeline(f"bufs_ptr = bufs_ptr_type({','.join(buf_ptrs)})")
        call_body.writeline("")

        if dipu_flag:
            call_body.writeline(
                f"dipu_stream = torch_dipu.current_stream({self.device_id}).dipu_stream"
            )

        call_str = 'kernel_cpp_0('
        if dipu_flag:
            call_str += 'c_void_p(dipu_stream), '
        else:
            call_str += 'c_void_p(), '
        call_str += "args_ptr, bufs_ptr)"
        call_body.writeline(call_str)

        if tops_check_precision:
            call_body.writeline("import sys")
            call_body.writeline(f"if '{self.folder}' not in sys.path:")
            with call_body.indent():
                call_body.writeline(f"sys.path.insert(0, '{self.folder}')")
            call_body.writeline(
                f"from {self.graph_key[:4]} import {self.graph_key} as graph_module")
            call_body.writeline("cpu_module = graph_module()")
            call_body.writeline(
                f"cpu_res = cpu_module({', '.join(map(lambda s: s + '.cpu()', args))})")
            call_body.writeline(
                f"check_res(list(cpu_res), list(({', '.join(map(lambda s: s + '.cpu()', bufs))},)), '{self.graph_key}')")

        for arg in args:
            if arg not in bufs:
                call_body.writeline(f'del {arg}')
        call_body.writeline("")

        if dipu_flag:
            call_body.writeline(f"torch_dipu.current_stream({self.device_id}).synchronize()")

        call_body.writeline(f"return ({', '.join(bufs[:len(bufs)-len(self.inplace_dict)])})")

        call_func = IndentedBuffer()
        call_func.writeline("def call(args):")
        with call_func.indent():
            call_func.splice(call_body)
        return call_func.getvalue()

    def gen_main_func(self):
        main_body = IndentedBuffer()
        main_body.splice(
            """
            from torch._dynamo.testing import rand_strided
            from torch._inductor.utils import print_performance
            """, strip=True
        )

        main_body.writeline("")
        for i in range(0, len(self.input_args)):
            itensor = self.input_args[i].meta['val']
            main_body.writeline('arg' + str(i) + ' = ' +
                                self.gen_random_tensor(itensor))

        args = []
        for i in range(len(self.input_args)):
            args.append('arg' + str(i))
        main_body.writeline("")
        main_body.writeline(f"print(call([{', '.join(args)}]))")

        main_func = IndentedBuffer()
        main_func.writeline("""if __name__ == "__main__":""")
        with main_func.indent():
            main_func.splice(main_body)
        return main_func.getvalue()

    def generate_code(self):
        return (self.gen_import_code() + self.gen_compile_graph_code() + self.gen_call_func() + self.gen_main_func())


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
                return arg
            elif isinstance(arg, torch.device):
                return str(arg)
            elif isinstance(arg, list):
                return [convert_arg(item) for item in arg]
            elif isinstance(arg, torch.layout):
                return
            elif isinstance(arg, torch.memory_format):
                return
            print(arg, type(arg))
            raise ValueError(f"unknown arg type({arg})")

        args_str = []
        kwargs_str = {}
        # name = process_name(node.name, node.target)
        # op_var = node.name

        for arg in args:
            if isinstance(arg, type(None)):
                continue
            if isinstance(arg, (list, tuple)):
                arg_list_str = []
                for elem in arg:
                    arg_list_str.append(convert_arg(elem))
                args_str.append(arg_list_str)
            else:
                args_str.append(convert_arg(arg))
        for k, v in kwargs.items():
            v = convert_arg(v)
            kwargs_str[k] = v

        return src_code, args_str, kwargs_str

    @staticmethod
    def Clone(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = {x};"

    @staticmethod
    def Copy(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = {y};"

    @staticmethod
    def Copy_(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = {y};"

    @staticmethod
    def LiftFreshCopy(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = {x};"

    @staticmethod
    def Abs(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Abs({x});"

    @staticmethod
    def make_const_if_scalar(op_var, value, dtype=torch.float32, count=0):
        src_code = ""
        if isinstance(value, numbers.Number):
            src_code = f"{cxx_type_set[dtype]} {op_var}_const_value{count} = static_cast<{cxx_type_set[dtype]}>({value});\n"
            value = f"{op_var}_const{count}"
            const_type = dtype if dtype != torch.float16 else torch.float32
            src_code += f"builder::Op {value} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_const_value{count}), builder::Type({{1}}, {type_set[const_type]}));\n"
            if dtype == torch.float16:
                src_code += f"{value} = builder::Convert({value}, builder::Type({{1}}, {type_set[dtype]}));\n"
        return src_code, value

    @staticmethod
    def make_type(op_var, dtype, shape=[1], count=0):
        dtype = type_set[dtype] if isinstance(dtype, torch.dtype) else dtype
        shape = f"{{{str(shape).split('[')[-1].split(']')[0]}}}"
        src_code = f"builder::Type {op_var}_type{count}({shape}, {dtype});\n"
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
    def Sub(op_var, shape, dtype, x, y, **kwargs_list):
        src_code_x, x = EnflameOverrides.make_const_if_scalar(op_var, x, dtype)
        src_code_y, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code = src_code_x + src_code_y
        src_code += f"builder::Op {op_var} = builder::Sub({x}, {y});"
        return src_code

    @staticmethod
    def Mul(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code += f"builder::Op {op_var} = builder::Mul({x}, {y});"
        return src_code

    @staticmethod
    def Dot(op_var, shape, dtype, x, y, **kwargs_list):
        src_code = f"builder::Op {op_var} = builder::Dot({x}, {y});\n"
        src_code += f"""{op_var}.SetAttribute("op_type", builder::Attribute("DotInference"));"""
        return src_code

    @staticmethod
    def DotGeneral(op_var, out_shape, out_dtype, lhs, rhs, lhs_batch_dims, rhs_batch_dims, lhs_contract_dims, rhs_contract_dims):
        lbd = '{' + ','.join([str(x) for x in lhs_batch_dims]) + '}'
        rbd = '{' + ','.join([str(x) for x in rhs_batch_dims]) + '}'
        lcd = '{' + ','.join([str(x) for x in lhs_contract_dims]) + '}'
        rcd = '{' + ','.join([str(x) for x in rhs_contract_dims]) + '}'
        dot_dimension_numbers_str = f"{lbd}, {rbd}, {lcd}, {rcd}"
        src_code = f"builder::DotDimensionNumbers {op_var}_dims_attr({dot_dimension_numbers_str});\n"
        src_code += f"builder::Op {op_var} = builder::DotGeneral({lhs}, {rhs}, {op_var}_dims_attr);\n"
        src_code += f"""{op_var}.SetAttribute("op_type", builder::Attribute("DotInference"));"""
        return src_code

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
    def NotEqual(op_var, shape, dtype, data_type, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(
            op_var, y, data_type)
        src_code += f"builder::Op {op_var} = builder::NotEqual({x}, {y});"
        return src_code

    @staticmethod
    def Log(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Log({x});"

    @staticmethod
    def Neg(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Neg({x});"

    @staticmethod
    def Pow(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(op_var, y, dtype)
        src_code += f"builder::Op {op_var} = builder::Pow({x}, {y});"
        return src_code

    @staticmethod
    def Exp(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Exp({x});"

    @staticmethod
    def Sqrt(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Sqrt({x});"

    @staticmethod
    def Relu(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Relu({x});"

    @staticmethod
    def Sigmoid(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Sigmoid({x});"

    @staticmethod
    def Reciprocal(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Reciprocal({x});"

    @staticmethod
    def Rsqrt(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::Rsqrt({x});"

    @staticmethod
    def Scalar(op_var, out_shape, out_dtype, x, **kwargs_list):
        src_code = f"{cxx_type_set[out_dtype]} {op_var}_value = static_cast<{cxx_type_set[out_dtype]}>({x});\n"
        src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, static_cast<void *>(&{op_var}_value), builder::Type({type_set[out_dtype]}));"
        return src_code

    @staticmethod
    def GetTupleElement(op_var, out_shape, out_dtype, t, idx):
        return f"builder::Op {op_var} = builder::GetTupleElement({t}, {int(idx)});"

    @staticmethod
    def NativeDropout(op_var, out_shape, out_dtype, *args):
        return f"builder::Op {op_var} = builder::Dropout({args[0]}, {args[1]}, {args[2]});\n"

    @staticmethod
    def MakeTuple(op_var, out_shape, out_dtype, *args):
        src_code = f"std::vector<builder::Op> {op_var}_outputs = {'{'} {', '.join(args)} {'}'};\n"
        src_code += f"builder::Op {op_var} = builder::Tuple({op_var}_outputs);\n"
        return src_code

    @staticmethod
    def Where(op_var, shape, dtype, condition, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::Select({condition}, {x}, {y});"

    @staticmethod
    def ZerosLike(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::ZerosLike({x}, {type_set[dtype]}, {{{str(shape).split('[')[-1].split(']')[0]}}});"

    @staticmethod
    def EmptyLike(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::EmptyLike({x}, {type_set[dtype]}, {{0}});"

    # TODO temporary fake implementation.
    @staticmethod
    def Bernoulli(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::OnesLike({x}, {type_set[dtype]}, {{{str(shape).split('[')[-1].split(']')[0]}}});"

    @staticmethod
    def NewEmptyStrided(op_var, shape, dtype, x, size, stride, **kwargs_list):
        return f"builder::Op {op_var} = builder::EmptyLike({x});"

    @staticmethod
    def OnesLike(op_var, out_shape, out_dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::OnesLike({x}, {type_set[out_dtype]}, {{{str(out_shape).split('[')[-1].split(']')[0]}}});"

    @staticmethod
    def Full(op_var, out_shape, out_dtype, size, value, **kwargs_list):
        src_code, op_type = EnflameOverrides.make_type(
            op_var, out_dtype, out_shape)
        src_code += f"builder::Op {op_var} = builder::Const(hlir_builder, {value}, {op_type});"
        return src_code

    @staticmethod
    def FullLike(op_var, shape, dtype, x, value, **kwargs_list):
        src_code = f"builder::Op {op_var} = builder::FullLike({x}, {value}, {type_set[dtype]}, {{{str(shape).split('[')[-1].split(']')[0]}}});"
        return src_code

    @staticmethod
    def Transpose(op_var, shape, dtype, x, permution=[0, 1], **kwargs_list):
        return f"builder::Op {op_var} = builder::Transpose({x}, {{{str(permution).strip('[]')}}});"

    @staticmethod
    def Hardswish(op_var, shape, dtype, x, **kwargs_list):
        return f"builder::Op {op_var} = builder::HardSwish({x});"

    @staticmethod
    def HardswishBackward(op_var, shape, dtype, x, y, **kwargs_list):
        return f"builder::Op {op_var} = builder::HardSwishGrad({x}, {y}, 3.0, 6.0, 6.0);"

    @staticmethod
    def Reshape(op_var, shape, dtype, x, new_size, **kwargs_list):
        src_code, op_type = EnflameOverrides.make_type(op_var, dtype, shape)
        src_code += f"builder::Op {op_var} = builder::Reshape({x}, {op_type});"
        return src_code

    @staticmethod
    def Expand(op_var, shape, dtype, x, new_shape, **kwargs_list):
        src_code, op_type = EnflameOverrides.make_type(
            op_var, dtype, new_shape)
        src_code += f"builder::Op {op_var} = builder::BroadcastInDim({x}, {{{', '.join(map(str, range(len(shape))))}}}, {op_type});"
        return src_code

    @staticmethod
    def Squeeze(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(
            op_var, y, torch.int64)
        src_code += f"builder::Op {op_var} = builder::Squeeze({x}, {y});"
        return src_code

    @staticmethod
    def Unsqueeze(op_var, shape, dtype, x, y, **kwargs_list):
        src_code, y = EnflameOverrides.make_const_if_scalar(
            op_var, y, torch.int64)
        src_code += f"builder::Op {op_var} = builder::Unsqueeze({x}, {y});"
        return src_code

    @staticmethod
    def ReduceMean(op_var, shape, dtype, x, axis="[]", keepdims="false", **kwargs_list):
        return f"builder::Op {op_var} = builder::ReduceMean({x}, {keepdims}, {{{str(axis).strip('[]')}}});"

    @staticmethod
    def ReduceMax(op_var, shape, dtype, x, axis="[]", keepdims="false", **kwargs_list):
        return f"builder::Op {op_var} = builder::ReduceMax({x}, {keepdims}, {{{str(axis).strip('[]')}}});"

    @staticmethod
    def ReduceSum(op_var, shape, dtype, x, axis="[]", keepdims="false", **kwargs_list):
        return f"builder::Op {op_var} = builder::ReduceSum({x}, {keepdims}, {{{str(axis).strip('[]')}}});"

    @staticmethod
    def Scatter(op_var, shape, dtype, x, dim, index, value, **kwargs_list):
        return f"auto {op_var} = enflame::Scatter(hlir_builder, {x}, {dim}, {index}, {value});"

    @staticmethod
    def Gather(op_var, shape, dtype, x, dim, index, **kwargs_list):
        src_code, op_type = EnflameOverrides.make_type(op_var, dtype, shape)
        src_code += f"auto {op_var} = enflame::Gather(hlir_builder, {x}, {index}, {dim}, {op_type});"
        return src_code

    @staticmethod
    def Slice(op_var, shape, dtype, start_indices, limit_indices, strides, x, *args, **kwargs_list):
        return f"builder::Op {op_var} = builder::Slice({x}, {{{', '.join(map(str, start_indices))}}}, "\
               f"{{{', '.join(map(str, limit_indices))}}}, {{{', '.join(map(str, strides))}}});"

    @staticmethod
    def SliceInDim(op_var, shape, dtype, x, dim, start, end, step, **kwargs_list):
        return f"builder::Op {op_var} = builder::SliceInDim({x}, {start}, {end}, {step}, {dim});"

    @staticmethod
    def SliceScatter(op_var, shape, dtype, x, y, dim, start, end, step, **kwargs_list):
        src_code_index, op_start_index = EnflameOverrides.make_const_if_scalar(
            op_var, 0, torch.int64, 0)
        src_code_index_dim, op_start_index_dim = EnflameOverrides.make_const_if_scalar(
            op_var, start, torch.int64, 1)
        src_code = src_code_index + src_code_index_dim
        src_code += f"builder::Op {op_var} = builder::DynamicUpdateSlice({x}, {y}, {{{', '.join([op_start_index_dim if i == dim else op_start_index for i in range(len(shape))])}}});"
        return src_code

    @staticmethod
    def BatchNorm(op_var, shape, dtype, input, weight, bias, running_mean, running_var, training, momentum, eps, **kwargs_list):
        return f"auto {op_var} = enflame::BatchNorm(hlir_builder, {input}, {weight}, {bias}, {running_mean}, {running_var}, 1, {training}, {momentum}, {eps});"

    @staticmethod
    def Conv2D(op_var, shape, dtype, inputs, *args, **kwargs_list):
        stride = f"{{{', '.join(map(str, args[len(inputs)]))}}}"
        padding = f"{{{args[len(inputs) + 1][0]}, {args[len(inputs) + 1][0]}, {args[len(inputs) + 1][1]}, {args[len(inputs) + 1][1]}}}"
        dilation = f"{{{', '.join(map(str, args[len(inputs) + 2]))}}}"
        return f'builder::Op {op_var} = builder::Conv2D({{{", ".join(inputs)}}}, 1, "NOTSET", "NCHW", {stride}, {padding}, {dilation});'

    @staticmethod
    def Conv2DBackward(op_var, shape, dtype, inputs, *args, **kwargs_list):
        bias_size = f"{{{', '.join(map(str, args[len(inputs)]))}}}"
        stride = f"{{{', '.join(map(str, args[len(inputs) + 1]))}}}"
        padding = f"{{{', '.join(map(str, args[len(inputs) + 2]))}}}"
        dilation = f"{{{', '.join(map(str,  args[len(inputs) + 3]))}}}"
        return f"auto {op_var} = enflame::Conv2D_Grad(hlir_builder, {', '.join(inputs)}, {bias_size}, {stride}, {padding}, {dilation});"

    @staticmethod
    def MaxPool2D(op_var, out_shape, out_dtype, shape, x, kernel_size, stride=[], padding=[0, 0], dilation=[1, 1], ceil_mode=False, **kwargs_list):
        ksize = f"{{{', '.join(map(str, kernel_size))}}}"
        stride = f"{{{', '.join(map(str, stride))}}}" if stride else f"{{{1, 1}}}"
        padding = f"{{{padding[0]}, {padding[0]}, {padding[1]}, {padding[1]}}}"
        shape = f"{{{', '.join(map(str, shape))}}}"
        return f"auto {op_var} = enflame::MaxPool2D(hlir_builder, {x}, {ksize}, {stride}, {padding}, {shape});"

    @staticmethod
    def MaxPool2DBackward(op_var, shape, dtype, *args):
        return f"auto {op_var} = enflame::MaxPool2D_Grad(hlir_builder, {args[0]}, {args[1]}, {{{', '.join(map(str, args[2]))}}}, {{{', '.join(map(str, args[3]))}}}, {{{', '.join(map(str, args[4]))}}});"

    @staticmethod
    def AvgPool2D(op_var, shape, dtype, reduce_dim, x, output_size, **kwargs):
        return f"builder::Op {op_var} = builder::ReduceMean({x}, true, {{{', '.join(map(str, reduce_dim))}}});"

    # [a + bi] ===> tops.tuple(a, bi)
    @staticmethod
    def ViewAsComplex(op_var, out_shape, out_dtype, x):
        shape = '{' + str(out_shape).split('[')[-1].split(']')[0] + '}'
        src_code = f"builder::Op {op_var} = enflame::ViewAsComplex(hlir_builder, {x}, {shape});"
        return src_code

    # tops.tuple(a, bi)====>[a,b]
    @staticmethod
    def ViewAsReal(op_var, out_shape, out_dtype, x):
        shape = '{' + str(out_shape).split('[')[-1].split(']')[0] + '}'
        return f"builder::Op {op_var} = enflame::ViewAsReal(hlir_builder, {x}, {shape});"

    # (a + bi)(c + di) = (ac -bd) + (ad + bd)i
    @staticmethod
    def ComplexMul(op_var, out_shape, out_dtype, x, y):
        return f"builder::Op {op_var} = enflame::ComplexMul(hlir_builder, {x}, {y});"

    @staticmethod
    def Concatenate(op_var, out_shape, out_dtype, tensors, dim):
        return f"builder::Op {op_var} = builder::Concatenate({'{' + ', '.join(tensors) + '}'}, {dim});"

    # Add an additional true flag for accuration in tops softmax.
    @staticmethod
    def Softmax(op_var, out_shape, out_dtype, x, y):
        return f"builder::Op {op_var} = builder::Softmax({x}, {y}, true);"

    @staticmethod
    def Logsoftmax(op_var, out_shape, out_dtype, x, y, z):
        return f"builder::Op {op_var} = builder::Softmax({x}, {y}, {z}, true);"

    @staticmethod
    def Gelu(op_var, out_shape, out_dtype, x, approximate, **kwargs):
        return f"builder::Op {op_var} = builder::Gelu({x}, {approximate});"

    @staticmethod
    def GeluBackward(op_var, out_shape, out_dtype, x, y, approximate):
        return f"builder::Op {op_var} = builder::GeluGrad({x}, {y}, {approximate});"

    @staticmethod
    def Iota(op_var, out_shape, out_dtype, length, **kwargs_list):
        src_code, op_type = EnflameOverrides.make_type(
            op_var, out_dtype, out_shape)
        src_code += f"builder::Op {op_var} = builder::Iota(hlir_builder, 0, {op_type});\n"
        return src_code

    @staticmethod
    def XlaGather(op_var, out_shape, out_dtype, operand, indices, offset_dims, collapsed_slice_dims,
                  start_index_map, index_vector_dim, slice_size):
        gather_dim_params = f"{op_var}_gather_dim_params"
        src_code = f"auto {gather_dim_params} = builder::GatherDimensionNumbers(\n" \
                   f"{'{'}{str(offset_dims)[1:-1]}{'}'}, {'{'}{str(collapsed_slice_dims)[1:-1]}{'}'}," \
                   f"{'{'}{str(start_index_map)[1:-1]}{'}'}, {index_vector_dim}\n" \
                   f");\n" \
                   f"builder::Op {op_var} = builder::Gather(\n" \
                   f"{operand}, {indices}, {gather_dim_params}, {'{'}{str(slice_size)[1:-1]}{'}'}\n" \
                   f");"

        return src_code
