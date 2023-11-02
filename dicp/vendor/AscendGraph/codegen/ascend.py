import json
import os
import uuid
import torch
from typing import Any, List
from torch.fx.node import Node
from torch._inductor.utils import IndentedBuffer
from dicp.vendor.AscendGraph.codegen.utils import (
    symint_in_shape,
    get_ascend_dtype,
    get_cpp_dtype,
    get_ascend_dtype_num
)

graph_id = 0

precision_check = bool(os.environ.get("DICP_ASCEND_PRECISION_CHECK", False))

sym_to_inputs = {}


def get_graph_id():
    global graph_id
    graph_id = graph_id + 1
    return graph_id


def process_name(name, target):
    if hasattr(target, "name"):
        real_op = target.name().split('::')[-1]
        if real_op.find('.') != -1:
            real_op = real_op.split('.')[0]
    else:
        real_op = name.rsplit('_', 1)[0] if name[-1].isdigit() else name
    return real_op


class AscendCodegen(torch.fx.Interpreter):
    def __init__(self, graph, aten_graph=None, folder=None, graph_key=None):
        self.graph = graph
        self.aten_graph = aten_graph
        self.override = AscendOverrides

        self.import_code = IndentedBuffer()
        self.build_graph_code = IndentedBuffer(initial_indent=1)

        self.graph_id = str(get_graph_id())
        self.args_dict = {}
        self.input_args = []
        self.output_args = []

        self.dynamic_inputs = []
        self.dynamic_shape = []
        self.actual_shape = []
        self.dynamic_index = []
        self.symint_outputs = []

        self.data_nodes = []
        self.common_nodes = []
        self.graph_input_names = []
        self.py_output_names = []
        self.graph_output_names = []
        self.build_options = []

        self.folder = folder
        self.graph_key = graph_key

        global sym_to_inputs
        sym_to_inputs = {}
        global sym_in_args
        sym_in_args = {}
        global arg_names
        arg_names = []

        # for modified args return
        global assign_args
        assign_args = []

        super().__init__(graph)

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = name
        self.input_args.append(self.cur_node)

        global arg_names
        arg_names = [arg.name for arg in self.input_args]
        fake_tensor = self.cur_node.meta['val']

        format = "NCHW"
        index = -1

        if isinstance(fake_tensor, torch.SymInt):
            dims = [1]
            data_type = "INT32"
            format = "ND"
            sym_to_inputs[fake_tensor.node.str()] = name
        elif symint_in_shape(fake_tensor.shape):
            # mention symint position in args
            for idx, dim in enumerate(fake_tensor.shape):
                if isinstance(dim, torch.SymInt):
                    st = dim.node.str()
                    if not st in sym_in_args:
                        sym_in_args[st] = (name, idx)

            # deal with dynamic shape -1
            shape = [-1 if isinstance(elem, torch.SymInt)
                     else elem for elem in fake_tensor.shape]
            actual_shape = [elem.node.str() if isinstance(
                elem, torch.SymInt) else str(elem) for elem in fake_tensor.shape]
            self.dynamic_inputs.append(self.args_dict[name])
            self.dynamic_shape.append(shape)
            self.actual_shape.append(actual_shape)
            self.dynamic_index.append(len(self.graph_input_names))
            dims = shape
            data_type = get_ascend_dtype(fake_tensor.dtype).upper()
        else:
            dims = list(fake_tensor.shape)
            data_type = get_ascend_dtype(fake_tensor.dtype).upper()

        if 'format' in self.cur_node.meta:
            format = self.cur_node.meta['format']
        # gen data_nodes
        self.data_nodes.append({
            "op_name": self.args_dict[name],
            "op_type": "Data",
            "dims": dims,
            "format": format,
            "data_type": data_type,
            "cpp_data_type": data_type,
            "index": index
        })
        self.graph_input_names.append(self.args_dict[name])

    def call_function(self, name, target, args, kwargs):
        if name not in self.args_dict.keys():
            self.args_dict[name] = name

        _, args_list = AscendOverrides.gen_args(
            self.args_dict[name], self.args_dict, args)
        real_op = process_name(name, target)
        op = getattr(self.override, real_op)(*args_list, **kwargs)
        if isinstance(op, list):
            self.common_nodes.extend(op)
        else:
            self.common_nodes.append(op)

    def get_attr(self, name, target, args, kwargs):
        assert isinstance(target, str)
        attr = self.fetch_attr(target)
        assert (isinstance(attr, torch.Tensor))
        self.args_dict[name] = name
        op = getattr(self.override, 'get_const_attr')(name, attr)
        self.common_nodes.append(op)

    def call_method(self, name, target, args, kwargs):
        pass

    def output(self, name, target, args, kwargs):
        for arg in args:
            self.output_args.extend(arg)

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
        return self.generate_code()

    def parse_outputs(self):
        symint_inputs = sym_to_inputs.values()
        for node in self.output_args:
            if isinstance(node, torch.fx.node.Node):
                name = self.args_dict[node.name]
                self.py_output_names.append(name)
                if name in self.graph_output_names:
                    continue
                else:
                    self.graph_output_names.append(name)
                if name in symint_inputs:
                    self.symint_outputs.append(name)
            else:
                self.py_output_names.append(str(node))

        if len(assign_args) > 0:
            self.graph_output_names.extend(list(zip(*assign_args))[0])

    def gen_import_code(self):
        self.import_code.splice(
            """
                from ctypes import c_void_p, c_long
                import torch
                import torch_dipu
                import random
                from torch import empty_strided, as_strided, device
                from dicp.dynamo_bridge.compile import AsyncCompileKernel
                from dicp.vendor.AscendGraph.compile_job import AscendCompileJob

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride

                def check_tensor(a, b, atol=5e-2, rtol=1e-2):
                    if not torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                        import pdb;pdb.set_trace()
                        pass
            """, strip=True
        )
        return self.import_code.getvalue()

    def process_sym_name(self, st):
        if st.isdigit():
            return st
        elif '+' in st:
            sp = st.split('+')
            assert (len(sp) == 2)
            sp = [elem.strip() for elem in sp]
            if sp[0] in sym_in_args:
                arg, idx = sym_in_args[sp[0]]
                return "{}.shape[{}]".format(arg, idx) + '+' + sp[1]
            return sym_to_inputs[sp[0]] + '+' + sp[1]
        elif '-' in st:
            sp = st.split('-')
            assert (len(sp) == 2)
            sp = [elem.strip() for elem in sp]
            if sp[0] in sym_in_args:
                arg, idx = sym_in_args[sp[0]]
                return "{}.shape[{}]".format(arg, idx) + '-' + sp[1]
            return sym_to_inputs[sp[0]] + '-' + sp[1]
        else:
            if st in sym_in_args:
                arg, idx = sym_in_args[st]
                return "{}.shape[{}]".format(arg, idx)
            return sym_to_inputs[st]

    def gen_call_func(self):
        # TODO check scalar input
        call_body = IndentedBuffer()
        self.args = [self.args_dict[x.name] for x in self.input_args]
        shape_symint = [value[0] for value in sym_in_args.values()]

        if len(sym_in_args) > 0 or len(sym_to_inputs) > 0:
            args = ['_' if not arg in shape_symint and not arg in sym_to_inputs.values() else arg for arg in self.args]
            call_body.writeline(f"({','.join(args)}) = args")

        # generate input dims
        if len(self.dynamic_inputs) > 0:
            dim_len = 0
            for shape in self.actual_shape:
                dim_len += len(shape)
            dims = f'''dims = {{'''
            for idx, elem in enumerate(self.actual_shape):
                if len(elem) == 0:
                    continue
                elem = [self.process_sym_name(dim) for dim in elem]
                dims += str(self.dynamic_index[idx]) + \
                    ":[" + ','.join(map(str, elem)) + '],'
            dims = dims[:-1] + f'''}}'''
            call_body.writeline(dims)
        else:
            call_body.writeline(f'''dims = None''')

        # generate output shapes
        if len(sym_in_args) > 0 or len(sym_to_inputs) > 0:
            shape_str = f'''output_shape = ['''
            for elem in self.output_args:
                if hasattr(elem, 'meta'):
                    elem = elem.meta['val']
                if isinstance(elem, torch.SymInt):
                    shape_str += '[1],'
                    continue
                shape = list(elem.shape)
                if len(shape) == 0:
                    raise RuntimeError("Error handling empty output_shape")
                shape = [self.process_sym_name(str(dim)) for dim in shape]
                shape_str += "[" + ','.join(map(str, shape)) + "],"

            # process output_shape with modified args
            for elem in assign_args:
                shape = list(self.input_args[elem[1]].meta['val'].shape)
                if len(shape) == 0:
                    raise RuntimeError("Error handling empty output_shape")
                shape = [self.process_sym_name(str(dim)) for dim in shape]
                shape_str += "[" + ','.join(map(str, shape)) + "],"
            shape_str = shape_str[:-1] + f''']'''
            call_body.writeline(shape_str)
        else:
            call_body.writeline(f'''output_shape = None''')

        call_body.splice("""
                             import torch_dipu
                             dipu_device_str = torch_dipu.dipu.device.__diputype__
                             args_new = []
                             for idx in range(len(args)):
                                 if isinstance(args[idx], int):
                                     args[idx] = torch.tensor(args[idx], device=dipu_device_str, dtype=torch.int32)
                                 if isinstance(args[idx], torch.Tensor):
                                     args_new.append(args[idx].clone())
                                     if not args_new[idx].is_contiguous():
                                         args_new[idx] = args_new[idx].contiguous()
                                     if args_new[idx].isnan().any() or args_new[idx].isinf().any() or args_new[idx].isneginf().any():
                                         args_new[idx] = torch.nan_to_num(args_new[idx])
                         """, strip=True)
        args_new = [f"{arg}_new" for arg in self.args]
        call_body.writeline(f"({','.join(args_new)}) = args_new")
        call_body.writeline(f"({','.join(self.args)}) = args")
        call_str = [f'output_tensor = kernel_cpp_0(args_new, dims, output_shape)']

        if precision_check and self.aten_graph is not None:
            # import aten graph
            call_str.append(f"import sys")
            call_str.append(f"if '{self.folder}' not in sys.path:")
            call_str.append(f"    sys.path.insert(0, '{self.folder}')")
            call_str.append(f"from {self.graph_key[:4]} import {self.graph_key} as graph_module")
            call_str.append(f"aten_call = graph_module()")

            call_str.append('aten_args = list(map(lambda x: x.to("cpu"), args))')
            call_str.append('for idx in modified:')
            call_str.append('    aten_args[idx] = aten_args[idx].item()')
            call_str.append('aten_output = aten_call(*aten_args)')

        for i, name in enumerate(self.graph_output_names):
            if name not in self.symint_outputs:
                call_str.append(f'{name} = output_tensor[{i}]')
            else:
                call_str.extend(f'del {name}',
                                 f'{name} = int(output_tensor[{i}])')

        # dealing with modified args passing back
        output_convert = [f'args[{name[1]}].copy_({name[0]})' for name in assign_args]
        del_args_new = [f'del ' + x for x in args_new]
        call_str.extend(del_args_new)
        call_str.append(f"args_new.clear()")
        call_str.extend(output_convert)
        del_args = [f'del ' + x for x in self.args if x not in self.py_output_names]
        call_str.extend(del_args)
        call_str.append(f"args.clear()")

        if precision_check:
            for i, name in enumerate(self.py_output_names):
                if name != 'None' and name not in self.args and name not in self.symint_outputs:
                    call_str.append(f"{name}_cpu = aten_output[{i}]")
                    call_str.append(f"check_tensor({name}.cpu(), {name}_cpu)")
        call_body.writelines(call_str)
        call_body.writeline(f"return ({', '.join(self.py_output_names)})")

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

        py_rand_inputs = []
        for i in range(len(self.input_args)):
            node = self.input_args[i]
            name = self.args[i]
            val = node.meta['val']
            if isinstance(val, torch.SymInt):
                code_str = f'''{name} = random.randint(0, 4)'''
            else:
                shape = str(tuple(val.size()))
                stride = str(tuple(val.stride()))
                device = val.device.type
                dtype = str(val.dtype)
                code_str = f'''{name} = rand_strided({shape}, {stride}, device='{device}', dtype={dtype})'''
            py_rand_inputs.append(code_str)
        main_body.writelines(py_rand_inputs)
        main_body.writeline(
            f"print_performance(lambda: call([{', '.join(self.args)}]))")

        main_func = IndentedBuffer()
        main_func.writeline("""if __name__ == "__main__":""")
        with main_func.indent():
            main_func.splice(main_body)
        return main_func.getvalue()

    def gen_build_options(self):
        if len(self.dynamic_inputs) > 0:
            self.build_options.append(
                {
                    "name": "input_format",
                    "value": "ND"
                }
            )
            value_str = ""
            for idx, name in enumerate(self.dynamic_inputs):
                value_str += f"{name}:"
                value_str += ','.join(map(str, self.dynamic_shape[idx])) + ';'
            value_str = value_str[:-1]
            self.build_options.append(
                {
                    "name": "input_shape",
                    "value": value_str
                }
            )

    def expand_symint(self, d, k):
        if isinstance(d[k], torch.SymInt):
            if d[k].node.str().isdigit():
                d[k] = d[k].node.hint
            else:
                import pdb
                pdb.set_trace()

    def remove_symint(self, cur):
        if isinstance(cur, list):
            for idx in range(len(cur)):
                self.expand_symint(cur, idx)
                self.remove_symint(cur[idx])
        elif isinstance(cur, dict):
            for k in cur.keys():
                self.expand_symint(cur, k)
                self.remove_symint(cur[k])

    def gen_graph_json(self):
        self.parse_outputs()
        self.gen_build_options()
        has_dynamic_shape = False if len(sym_in_args) == 0 and len(sym_to_inputs) == 0 else True
        graph = {
            "name": "graph",
            "input_names": self.graph_input_names,
            "output_names": self.graph_output_names,
            "has_dynamic_shape": has_dynamic_shape,
            "build_options": self.build_options,
            "data_nodes": self.data_nodes,
            "common_nodes": self.common_nodes,
        }
        self.remove_symint(graph)
        return json.dumps(graph)

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        graph_json = self.gen_graph_json()
        compile_graph_code.splice(
            f"""
                ascend_compile_job = AscendCompileJob('''{graph_json}''')
                async_compile = AsyncCompileKernel()
                kernel_cpp_0 = async_compile.compile_kernel(ascend_compile_job)
            """, strip=True
        )
        compile_graph_code.writeline('async_compile.wait(globals())')
        compile_graph_code.writeline('del async_compile')
        return compile_graph_code.getvalue()

    def generate_code(self):
        return (self.gen_import_code() + self.gen_compile_graph_code() + self.gen_call_func() + self.gen_main_func())


class AscendOperator:
    def __init__(self, op_name: str, op_type: str):
        self.op_name = op_name
        self.op_type = op_type
        self.inputs = []
        self.attrs = []
        self.dynamic_inputs = []
        self.dynamic_outputs = []

    def to_node(self):
        node = {
            "op_name": self.op_name,
            "op_type": self.op_type,
        }
        if len(self.inputs) > 0:
            node["inputs"] = self.inputs
        if len(self.attrs) > 0:
            node["attrs"] = self.attrs
        if len(self.dynamic_inputs) > 0:
            node["dynamic_inputs"] = self.dynamic_inputs
        if len(self.dynamic_outputs) > 0:
            node["dynamic_outputs"] = self.dynamic_outputs
        return node

    def set_input(self, name, value):
        self.inputs.append({
            "name": name,
            "value": value,
        })

    def set_input_with_index(self, name, value, index):
        self.inputs.append({
            "name": name,
            "value": value,
            "index": index,
        })

    def set_dynamic_output(self, name, num):
        self.dynamic_outputs.append({
            "name": name,
            "num": num
        })

    def set_and_update_input(self, name, value, shape, format, data_type, output_name="y"):
        self.inputs.append({
            "name": name,
            "value": value,
            "update_desc": {
                "format": format,
                "shape": shape,
                "data_type": data_type,
                "output_name": output_name,
            }
        })

    def set_dynamic_input(self, name, num, value, set_input_with_name=False):
        assert len(value) == num
        dy_inputs = {
            "name": name,
            "num": num,
            "value": [],
        }
        if set_input_with_name is False:
            for i in range(num):
                dy_inputs["value"].append({
                    "index": i,
                    "value": value[i],
                })
        else:
            for i in range(num):
                dy_inputs["value"].append({
                    "index": i,
                    "value": value[i]["input_name"],
                    "edge": value[i]["edge_name"]
                })
        self.dynamic_inputs.append(dy_inputs)

    def set_attr_list_int(self, name: str, value: List[int]):
        self.attrs.append({
            "name": name,
            "value_type": "list_int",
            "value": value,
        })

    def set_attr_list_float(self, name: str, value: List[float]):
        self.attrs.append({
            "name": name,
            "value_type": "list_float",
            "value": value,
        })

    def set_attr_bool(self, name: str, value: bool):
        self.attrs.append({
            "name": name,
            "value_type": "bool",
            "value": value,
        })

    def set_attr_str(self, name: str, value: str):
        self.attrs.append({
            "name": name,
            "value_type": "str",
            "value": value,
        })

    def set_attr_int(self, name: str, value: int):
        self.attrs.append({
            "name": name,
            "value_type": "int",
            "value": value
        })

    def set_attr_int64(self, name: str, value: int):
        self.attrs.append({
            "name": name,
            "value_type": "int64",
            "value": value
        })

    def set_attr_float(self, name: str, value: float):
        self.attrs.append({
            "name": name,
            "value_type": "float",
            "value": float(value)
        })

    def set_attr_dtype_str(self, name: str, value: str):
        self.attrs.append({
            "name": name,
            "value_type": "dtype_str",
            "value": value
        })

    def set_attr_tensor(self, name: str, data_type: str,
                        cpp_data_type: str,
                        format: str,
                        value: List,
                        dims: List[int]):
        self.attrs.append({
            "name": name,
            "value_type": "tensor",
            "tensor_data_type": data_type,
            "tensor_cpp_data_type": cpp_data_type,
            "tensor_format": format,
            "tensor_value": value,
            "tensor_dims": dims,
        })


OP = AscendOperator


class AscendOverrides:
    @staticmethod
    def gen_args(op_var, args_dict, args):
        src_code = IndentedBuffer()
        args_str = [op_var]
        for i in range(len(args)):
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            else:
                args_str.append(args[i])
        return src_code, args_str

    @staticmethod
    def Mul(name, x, y):
        op = OP(name, "Mul")
        op.set_input("x1", x)
        op.set_input("x2", y)
        return op.to_node()

    @staticmethod
    def IdentityN(name, *args, **kwargs):
        input_names = []
        for input in args:
            if f"{input}_edge_name" in kwargs:
                edges: list(str) = kwargs[f"{input}_edge_name"]
                for i in range(len(edges)):
                    input_names.append(
                        {"input_name": input, "edge_name": edges[i]})
            else:
                input_names.append(input)
        id_op = OP(name, "IdentityN")
        id_op.set_dynamic_input("x", len(input_names),
                                input_names, len(kwargs) > 0)
        id_op.set_dynamic_output("y", len(input_names))
        return id_op.to_node()

    @staticmethod
    def adds(name, x, y):
        adds_op = OP(name, "Adds")
        adds_op.set_input("x", x)
        adds_op.set_attr_float("value", float(y))
        return adds_op.to_node()

    @staticmethod
    def add(name, x, y):
        add_op = OP(name, "Add")
        add_op.set_input("x1", x)
        add_op.set_input("x2", y)
        return add_op.to_node()

    @staticmethod
    def Sub(name, x, y):
        sub_op = OP(name, "Sub")
        sub_op.set_input("x1", x)
        sub_op.set_input("x2", y)
        return sub_op.to_node()

    @staticmethod
    def Relu(name, x):
        op = OP(name, "Relu")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Swish(name, x, scale):
        silu_op = OP(name, "Swish")
        silu_op.set_input("x", x)
        silu_op.set_attr_float("scale", scale)
        return silu_op.to_node()

    @staticmethod
    def Transpose(name, input, perm):
        transpose_op = OP(name, "Transpose")
        transpose_op.set_input("x", input)
        transpose_op.set_input("perm", perm)
        return transpose_op.to_node()

    @staticmethod
    def reciprocal(name, x):
        op = OP(name, "Reciprocal")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Sqrt(name, x):
        op = OP(name, "Sqrt")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Div(name, x1, x2):
        op = OP(name, "Div")
        op.set_input("x1", x1)
        op.set_input("x2", x2)
        return op.to_node()

    @staticmethod
    def DivNoNan(name, x1, x2):
        op = OP(name, "DivNoNan")
        op.set_input("x1", x1)
        op.set_input("x2", x2)
        return op.to_node()

    @staticmethod
    def Select(name, cond, x1, x2):
        op = OP(name, "SelectV2")
        op.set_input("condition", cond)
        op.set_input("then", x1)
        op.set_input("else", x2)
        return op.to_node()

    @staticmethod
    def Rsqrt(name, x):
        op = OP(name, "Rsqrt")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Conv2D(name, input, weight, stride, padding,
               dilation, groups, format, bias):
        op = OP(name, "Conv2D")
        op.set_input("x", input)
        op.set_input("filter", weight)
        op.set_attr_list_int("strides", stride)
        op.set_attr_list_int("pads", padding)
        op.set_attr_list_int("dilations", dilation)
        op.set_attr_int("groups", groups)
        op.set_attr_str("data_format", format)
        if bias is not None:
            op.set_input("bias", bias)
        return op.to_node()

    @staticmethod
    def ReduceMean(name, x, axes, keepdim=False):
        mean_op = OP(name, "ReduceMean")
        mean_op.set_input("x", x)
        mean_op.set_input("axes", axes)
        mean_op.set_attr_bool("keep_dims", keepdim)
        return mean_op.to_node()

    @staticmethod
    def GreaterEqual(name, x, y):
        ge_op = OP(name, "GreaterEqual")
        ge_op.set_input("x1", x)
        ge_op.set_input("x2", y)
        return ge_op.to_node()

    @staticmethod
    def AddV2(name, x1, x2):
        add_op = OP(name, "AddV2")
        add_op.set_input("x1", x1)
        add_op.set_input("x2", x2)
        return add_op.to_node()

    @staticmethod
    def get_const_attr(name, x):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        x_shape = list(x.shape)
        x_value = x.tolist()
        if not isinstance(x_value, list):
            x_value = [x_value]

        torch_dtype = x.dtype
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)

        op = OP(name, "Const")
        op.set_attr_tensor("value", ascend_dtype, cpp_dtype,
                           "ND", x_value, x_shape)

        return op.to_node()

    @staticmethod
    def MaskedFill(name, x, mask, value):
        op = OP(name, "MaskedFill")
        op.set_input("x", x)
        op.set_input("mask", mask)
        op.set_input("value", value)
        return op.to_node()

    @staticmethod
    def Unsqueeze(name, x, dim):
        op = OP(name, "Unsqueeze")
        op.set_input("x", x)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def Squeeze(name, x, dim):
        op = OP(name, "Squeeze")
        op.set_input("x", x)
        op.set_attr_list_int("axis", dim)
        return op.to_node()

    @staticmethod
    def Identity(name, input, index):
        op = OP(name, "Identity")
        if index is not None:
            if isinstance(index, int):
                op.set_input_with_index("x", input, index)
            else:
                if index in arg_names:
                    assign_args.append((name, arg_names.index(index)))
                op.set_input("x", input)
        else:
            op.set_input("x", input)
        return op.to_node()

    @staticmethod
    def Exp(name, x):
        op = OP(name, "Exp")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Sigmoid(name, x):
        op = OP(name, "Sigmoid")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Pow(name, x, exp):
        op = OP(name, "Pow")
        op.set_input("x1", x)
        op.set_input("x2", exp)
        return op.to_node()

    @staticmethod
    def Maximum(name, a, b):
        op = OP(name, "Maximum")
        op.set_input("x1", a)
        op.set_input("x2", b)
        return op.to_node()

    @staticmethod
    def SoftmaxV2(name, x, dim):
        op = OP(name, "SoftmaxV2")
        op.set_input("x", x)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def ReduceSumD(name, x, axes, keep_dims):
        op = OP(name, "ReduceSumD")
        op.set_input("x", x)
        op.set_attr_list_int("axes", axes)
        op.set_attr_bool("keep_dims", keep_dims)
        return op.to_node()

    @staticmethod
    def ReduceMaxD(name, x, axes, keep_dims):
        op = OP(name, "ReduceMaxD")
        op.set_input("x", x)
        op.set_attr_list_int("axes", axes)
        op.set_attr_bool("keep_dims", keep_dims)
        return op.to_node()

    @staticmethod
    def Permute(name, x, order=[0]):
        op = OP(name, "Permute")
        op.set_input("x", x)
        op.set_attr_list_int("order", order)
        return op.to_node()

    @staticmethod
    def ReduceStdV2Update(name, x, mean, dim, unbiased, keepdim):
        op = OP(name, "ReduceStdV2Update")
        op.set_input("x", x)
        op.set_input("mean", mean)
        op.set_attr_list_int("dim", dim)
        op.set_attr_bool("unbiased", unbiased)
        op.set_attr_bool("keepdim", keepdim)
        return op.to_node()

    @staticmethod
    def Log(name, x):
        op = OP(name, "Log")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Neg(name, x):
        op = OP(name, "Neg")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Expand(name, x, shape):
        op = OP(name, "Expand")
        op.set_input("x", x)
        op.set_input("shape", shape)
        return op.to_node()

    @staticmethod
    def ExpandD(name, x, shape):
        op = OP(name, "ExpandD")
        op.set_input("x", x)
        op.set_attr_list_int("shape", shape)
        return op.to_node()

    @staticmethod
    def ZerosLike(name, x, *args):
        # TODO(tangzhiyi): ignore kwargs, need to check this
        op = OP(name, "ZerosLike")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Fill(name, dims, value):
        op = OP(f"{name}", "Fill")
        op.set_input("dims", dims)
        op.set_input("value", value)
        return op.to_node()

    @staticmethod
    def Cast(name, x, ascend_dtype):
        cast_op = OP(name, "Cast")
        cast_op.set_input("x", x)
        cast_op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
        return cast_op.to_node()

    @staticmethod
    def Const(name, x, dtype, dims=None, format="ND"):
        if not isinstance(x, list):
            x = [x]
        assert len(x) > 0
        ascend_dtype = get_ascend_dtype(dtype)
        cpp_dtype = get_cpp_dtype(dtype)
        const_op = OP(name, "Const")
        const_op.set_attr_tensor(
            "value", ascend_dtype, cpp_dtype, format, x, [len(x)] if dims is None else dims)
        return const_op.to_node()

    @staticmethod
    def BroadcastTo(name, x, shape):
        broadcast_op = OP(name, "BroadcastTo")
        broadcast_op.set_input("x", x)
        broadcast_op.set_input("shape", shape)
        return broadcast_op.to_node()

    @staticmethod
    def Empty(name, shape, dtype, layout=torch.strided, device='cpu'):
        dtype = get_ascend_dtype_num(get_ascend_dtype(dtype))
        op = OP(f"{name}", "Empty")
        op.set_input("shape", shape)
        op.set_attr_int("dtype", dtype)
        return op.to_node()

    @staticmethod
    def OnesLike(name, x):
        op = OP(name, "OnesLike")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def Sort(name, x, dim, descending):
        op = OP(name, "Sort")
        op.set_input("x", x)
        op.set_attr_int("axis", dim)
        op.set_attr_bool("descending", descending)
        op.set_attr_int("_keep_dtype", 1)
        return op.to_node()

    @staticmethod
    def TopK(name, x, k, dim, largest, sorted):
        op = OP(name, "TopK")
        op.set_input("x", x)
        op.set_input("k", k)
        op.set_attr_int("dim", dim)
        op.set_attr_bool("largest", largest)
        op.set_attr_bool("sorted", sorted)
        op.set_attr_int("_keep_dtype", 1)
        return op.to_node()

    @staticmethod
    def ScatterElements(name, var, index, value, dim):
        op = OP(name, "ScatterElements")
        op.set_input("data", var)
        op.set_input("indices", index)
        op.set_input("updates", value)
        op.set_attr_int("axis", dim)
        return op.to_node()

    @staticmethod
    def MatMul(name, x1, x2, trans_x1: bool, trans_x2: bool):
        # TODO! MatMul not support fp32 input
        # for higher precision
        cast_op1 = OP(f'{name}_x1_cast', "Unsqueeze")
        cast_op1.set_input('x', x1)
        cast_op1.set_attr_list_int("axes", [0])

        cast_op2 = OP(f'{name}_x2_cast', "Unsqueeze")
        cast_op2.set_input('x', x2)
        cast_op2.set_attr_list_int("axes", [0])

        op = OP(f'{name}_matmul', "BatchMatMul")
        op.set_input("x1", f'{name}_x1_cast')
        op.set_input("x2", f'{name}_x2_cast')
        op.set_attr_bool("adj_x1", trans_x1)
        op.set_attr_bool("adj_x2", trans_x2)
        op.set_attr_int("_keep_dtype", 1)

        res_op = OP(name, "Squeeze")
        res_op.set_input("x", f'{name}_matmul')
        res_op.set_attr_list_int("axis", [0])
        return [cast_op1.to_node(), cast_op2.to_node(), op.to_node(), res_op.to_node()]

    @staticmethod
    def BatchMatMul(name, x1, x2, adj_x1: bool, adj_x2: bool):
        op = OP(name, "BatchMatMul")
        op.set_input("x1", x1)
        op.set_attr_bool("adj_x1", adj_x1)
        op.set_input("x2", x2)
        op.set_attr_bool("adj_x2", adj_x2)
        op.set_attr_int("_keep_dtype", 1)
        return op.to_node()

    @staticmethod
    def Conv2DBackpropInput(name, input_size, filter, out_backprop, strides, pads,
                            dilations, groups, data_format):
        bp_op = OP(name, "Conv2DBackpropInput")
        bp_op.set_input("input_size", input_size)
        bp_op.set_input("filter", filter)
        bp_op.set_input("out_backprop", out_backprop)
        bp_op.set_attr_list_int("strides", strides)
        bp_op.set_attr_list_int("pads", pads)
        bp_op.set_attr_list_int("dilations", dilations)
        bp_op.set_attr_int("groups", groups)
        bp_op.set_attr_str("data_format", data_format)
        return bp_op.to_node()

    @staticmethod
    def Conv2DBackpropFilter(name, x, filter_size, out_backprop, strides, pads,
                             dilations, data_format):
        bp_op = OP(name, "Conv2DBackpropFilter")
        bp_op.set_input("x", x)
        bp_op.set_input("filter_size", filter_size)
        bp_op.set_input("out_backprop", out_backprop)
        bp_op.set_attr_list_int("strides", strides)
        bp_op.set_attr_list_int("pads", pads)
        bp_op.set_attr_list_int("dilations", dilations)
        bp_op.set_attr_str("data_format", data_format)
        return bp_op.to_node()

    @staticmethod
    def PadV3(name, x, paddings):
        op = OP(name, "PadV3")
        op.set_input("x", x)
        op.set_input("paddings", paddings)
        return op.to_node()

    @staticmethod
    def PadV3Grad(name, x, paddings):
        pad_grad = OP(name, "PadV3Grad")
        pad_grad.set_input("x", x)
        pad_grad.set_input("paddings", paddings)

    @staticmethod
    def MaxPool(name, x, ksize, strdes, padding, data_format):
        fwd_out = OP(name, "MaxPool")
        fwd_out.set_input("x", x)
        fwd_out.set_attr_list_int("ksize", ksize)
        fwd_out.set_attr_list_int("strides", strdes)
        fwd_out.set_attr_str("padding", padding)
        fwd_out.set_attr_str("data_format", data_format)
        return fwd_out.to_node()

    @staticmethod
    def MaxPoolGrad(name, x1, x2, grad, ksize, strdes, padding, data_format):
        bwd = OP(name, "MaxPoolGrad")
        bwd.set_input("x1", x1)
        bwd.set_input("x2", x2)
        bwd.set_input("grad", grad)
        bwd.set_attr_list_int("ksize", ksize)
        bwd.set_attr_list_int("strides", strdes)
        bwd.set_attr_str("padding", padding)
        bwd.set_attr_str("data_format", data_format)
        return bwd.to_node()

    @staticmethod
    def LessEqual(name, x1, x2):
        op = OP(name, "LessEqual")
        op.set_input("x1", x1)
        op.set_input("x2", x2)
        return op.to_node()

    @staticmethod
    def Less(name, x1, x2):
        cond_op = OP(name, "Less")
        cond_op.set_input("x1", x1)
        cond_op.set_input("x2", x2)
        return cond_op.to_node()

    @staticmethod
    def ret_tuple(name, in1, in2):
        op = OP(name, "IdentityN")
        op.set_dynamic_input("x", 2, [in1, in2])
        op.set_dynamic_output("y", 2)
        return op.to_node()

    @staticmethod
    def ret_triple(name, in1, in2, in3):
        op = OP(name, "IdentityN")
        op.set_dynamic_input("x", 3, [in1, in2, in3])
        op.set_dynamic_output("y", 3)
        return op.to_node()

    @staticmethod
    def Range(name, end, start, step):
        op = OP(name, "Range")
        op.set_input("start", start)
        op.set_input("limit", end)
        op.set_input("delta", step)
        return op.to_node()

    @staticmethod
    def Equal(name, a, b):
        eq_op = OP(name, "Equal")
        eq_op.set_input("x1", a)
        eq_op.set_input("x2", b)
        return eq_op.to_node()

    @staticmethod
    def Cumsum(name, x, dim):
        op1 = OP(name, "Cumsum")
        op1.set_input("x", x)
        op1.set_input("axis", dim)
        return op1.to_node()

    @staticmethod
    def LogSoftmaxV2(name, x, dim):
        op = OP(name, "LogSoftmaxV2")
        op.set_input("logits", x)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def LogSoftmaxGrad(name, grad_output, x, dim):
        op = OP(name, "LogSoftmaxGrad")
        op.set_input("grad", grad_output)
        op.set_input("x", x)
        op.set_attr_list_int("axis", dim)
        return op.to_node()

    @staticmethod
    def BNTrainingReduce(name, x, x_shape, format, dtype):
        dtype = get_ascend_dtype(dtype)
        op = OP(name, "BNTrainingReduce")
        # TODO(tangzhiyi): now assume output name is y.
        # TODO(daoxin): potential dynamic shape issue in resnet18
        op.set_and_update_input("x", x, x_shape, format, dtype)
        return op.to_node()

    @staticmethod
    def BNTrainingUpdate(name, x, sum, sum_idx, square_sum, square_idx, weight,
                         bias, running_mean, running_var, eps, momentum):
        op = OP(name, "BNTrainingUpdate")
        op.set_input("x", x)
        op.set_input_with_index("sum", sum, sum_idx)
        op.set_input_with_index("square_sum", square_sum, square_idx)
        op.set_input("scale", weight)
        op.set_input("offset", bias)
        op.set_input("mean", running_mean)
        op.set_input("variance", running_var)
        op.set_attr_float("epsilon", eps)
        op.set_attr_float("factor", momentum)
        return op.to_node()

    @staticmethod
    def BNTrainingUpdateGrad(name, grad_out, x_shape, format, x_dtype, backprops,
                             x, save_mean, save_invstd, eps):
        x_dtype = get_ascend_dtype(x_dtype)
        op = OP(name, "BNTrainingUpdateGrad")
        op.set_and_update_input(
            "grads", grad_out, x_shape, format, x_dtype, backprops)
        op.set_input("x", x)
        op.set_input("batch_mean", save_mean)
        op.set_input("batch_variance", save_invstd)
        op.set_attr_float("epsilon", eps)
        return op.to_node()

    @staticmethod
    def BNTrainingReduceGrad(name, grad_out, x, scale, scale_idx, offset, offset_idx,
                             weight, save_mean, save_invstd, eps):
        op = OP(name, "BNTrainingReduceGrad")
        op.set_input("grads", grad_out)
        op.set_input("x", x)
        op.set_input_with_index("diff_scale", scale, scale_idx)
        op.set_input_with_index("diff_offset", offset, offset_idx)
        op.set_input("scale", weight)
        op.set_input("batch_mean", save_mean)
        op.set_input("batch_variance", save_invstd)
        op.set_attr_float("epsilon", eps)
        return op.to_node()

    @staticmethod
    def FillV2D(name, value, dims):
        op = OP(name, "FillV2D")
        op.set_attr_float("value", value)
        op.set_attr_list_int("dims", dims)
        return op.to_node()

    @staticmethod
    def NLLLoss(name, x, target, weight, reduction, ignore_index):
        op = OP(name, "NLLLoss")
        op.set_input("x", x)
        op.set_input("target", target)
        op.set_input("weight", weight)
        op.set_attr_str("reduction", reduction)
        op.set_attr_int("ignore_index", ignore_index)
        return op.to_node()

    @staticmethod
    def NLLLossGrad(name, x, y_grad, target, weight, total_weight, reduction, ignore_index):
        op = OP(name, "NLLLossGrad")
        op.set_input("x", x)
        op.set_input("y_grad", y_grad)
        op.set_input("target", target)
        op.set_input("weight", weight)
        op.set_input("total_weight", total_weight)
        op.set_attr_str("reduction", reduction)
        op.set_attr_int("ignore_index", ignore_index)
        return op.to_node()

    @staticmethod
    def ReluGrad(name, grad_output, x):
        op = OP(name, "ReluGrad")
        op.set_input("gradients", grad_output)
        op.set_input("features", x)
        return op.to_node()

    @staticmethod
    def ThresholdGradV2D(name, grad_output, x, threshold):
        op = OP(name, "ThresholdGradV2D")
        op.set_input("gradients", grad_output)
        op.set_input("features", x)
        op.set_attr_float("threshold", threshold)
        return op.to_node()

    @staticmethod
    def SplitD(name, x, dim, num_split, y):
        split_op = OP(name, "SplitD")
        split_op.set_input("x", x)
        split_op.set_attr_int("split_dim", dim)
        split_op.set_attr_int("num_split", num_split)
        split_op.set_dynamic_output("y", y)
        return split_op.to_node()

    @staticmethod
    def Pack(name, x, axis):
        x = [elem.name for elem in x]
        op = OP(f"{name}", "Pack")
        op.set_dynamic_input("x", len(x), x)
        op.set_attr_int("axis", axis)
        op.set_attr_int("N", len(x))
        return op.to_node()

    @staticmethod
    def Slice(name, x, offsets, size):
        # TODO(tangzhiyi): miss step parameter
        op = OP(name, "Slice")
        op.set_input("x", x)
        op.set_input("offsets", offsets)
        op.set_input("size", size)
        return op.to_node()

    @staticmethod
    def ConcatD(name, x, dim):
        x_list = [a.name if isinstance(
            a, torch.fx.node.Node) else a for a in x]
        op = OP(name, "ConcatD")
        op.set_dynamic_input("x", len(x), x_list)
        op.set_attr_int("N", len(x))
        op.set_attr_int("concat_dim", dim)
        return op.to_node()

    @staticmethod
    def Reshape(name, x, shape):
        op = OP(name, "Reshape")
        op.set_input("x", x)
        op.set_input("shape", shape)
        return op.to_node()

    @staticmethod
    def GatherV2(name, x, indices, axis):
        gather_op = OP(name, "GatherV2")
        gather_op.set_input("x", x)
        gather_op.set_input("indices", indices)
        gather_op.set_input("axis", axis)
        return gather_op.to_node()

    @staticmethod
    def Pad(name, x, paddings):
        pad_op = OP(name, "Pad")
        pad_op.set_input("x", x)
        pad_op.set_input("paddings", paddings)
        return pad_op.to_node()

    @staticmethod
    def Fills(name, x, value):
        fills_op = OP(name, "Fills")
        fills_op.set_input("x", x)
        fills_op.set_attr_float("value", float(value))
        return fills_op.to_node()

    @staticmethod
    def SoftmaxGrad(name, grad_output, output, dim):
        op = OP(name, "SoftmaxGrad")
        op.set_input("grad_softmax", grad_output)
        op.set_input("softmax", output)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def StatelessBernoulli(name, shape, prop, seed, offset, dtype):
        dtype = get_ascend_dtype(dtype)
        bernoulli_op = OP(name, "StatelessBernoulli")
        bernoulli_op.set_input("shape", shape)
        bernoulli_op.set_input("prob", prop)
        bernoulli_op.set_input("seed", seed)
        bernoulli_op.set_input("offset", offset)
        bernoulli_op.set_attr_dtype_str("dtype", dtype)
        return bernoulli_op.to_node()

    @staticmethod
    def Shape(name, x):
        shape_op = OP(name, "Shape")
        shape_op.set_input("x", x)
        return shape_op.to_node()

    @staticmethod
    def StatelessRandomUniformV2(name, shape, key, counter, alg, ascend_dtype):
        rand_op = OP(name, "StatelessRandomUniformV2")
        rand_op.set_input("shape", shape)
        rand_op.set_input("key", key)
        rand_op.set_input("counter", counter)
        rand_op.set_input("alg", alg)
        rand_op.set_attr_dtype_str("dtype", ascend_dtype)
        return rand_op.to_node()

    @staticmethod
    def Greater(name, x1, x2):
        op = OP(name, "Greater")
        op.set_input("x1", x1)
        op.set_input("x2", x2)
        return op.to_node()

    @staticmethod
    def Addcmul(name, input_data, x1, x2, value):
        op = OP(name, "Addcmul")
        op.set_input("input_data", input_data)
        op.set_input("x1", x1)
        op.set_input("x2", x2)
        op.set_input("value", value)
        return op.to_node()

    @staticmethod
    def Reciprocal(name, x):
        op = OP(name, "Reciprocal")
        op.set_input("x", x)
        return op.to_node()
