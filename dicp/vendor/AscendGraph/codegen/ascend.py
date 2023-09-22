import json
import os
import random
import uuid

import numpy as np
import torch

from typing import Any, List
from torch.fx.node import Node
from torch._inductor.utils import IndentedBuffer

graph_id = 0

precision_check = bool(os.environ.get("DICP_ASCEND_PRECISION_CHECK", False))

need_node = ['add', 'mul', 'div', 'view', 'scatter', 'full', 'lt', 'inge', 'eq',
             'where', 'convolution', 'le', 'scalar_tensor', 'maximum', 'inadd',
             't', 'nll_loss_forward', 'native_batch_norm_legit_functional', 'gather',
             'nll_loss_backward', 'native_batch_norm_backward', 'repeat_interleave',
             'view_as_complex', 'view_as_real', 'slice', 'select', 'topk', 'sub',
             'pow', 'cat', 'expand', 'transpose', 'inmul', 'mm', 'masked_fill',
             'rsub', 'index', 'slice_backward', 'empty_like', 'fill_scalar',
             'bernoulli', 'new_empty_strided', 'fill']

sym_to_inputs = {}
def get_graph_id():
    global graph_id
    graph_id = graph_id + 1
    return graph_id


def get_reduction_str(r):
    if r == 0:
        return "none"
    elif r == 1:
        return "mean"
    elif r == 2:
        return "sum"
    else:
        raise RuntimeError("not supported yet!")
      

def process_name(name, target):
    if hasattr(target, "name"):
        real_op = target.name().split('::')[-1]
        if real_op.find('.') != -1:
            real_op = real_op.split('.')[0]
    else:
        real_op = name.rsplit('_', 1)[0] if name[-1].isdigit() else name
    return real_op


def process_dynamic_shape(shape, name, suffix = "preprocess"):
    ops = []
    x_names = []

    def generate_digits_op(shapes):
        count = len(x_names)
        op = OP(f"{name}_dim_{suffix}{count}", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", shapes, [len(shapes)])
        ops.append(op.to_node())
        x_names.append(f"{name}_dim_{suffix}{count}")

    def generate_sym_int(elem):
        count = len(x_names)
        elem = elem.node.str()
        elems = elem.strip().split(' ')

        if len(elems) > 1:
            assert len(elems) == 3
            assert elems[2].isdigit()
            assert elems[1] == '+' or elems[1] == '-'
            op_type = "Add" if elems[1] == '+' else "Sub"
            op1 = OP(f"{name}_dim_{suffix}{count}_const", "Const")
            op1.set_attr_tensor("value", "INT32", "INT32", "ND", [int(elems[2])], [1])
            op2 = OP(f"{name}_dim_{suffix}{count}", op_type)
            op2.set_input("x1", sym_to_inputs[elems[0]])
            op2.set_input("x2", f"{name}_dim_{suffix}{count}_const")
            ops.extend([op1.to_node(), op2.to_node()])
            x_names.append(f"{name}_dim_{suffix}{count}")
        else:
            x_names.append(sym_to_inputs[elems[0]])


    dims = []
    for elem in shape:
        if not isinstance(elem, torch.SymInt):
            dims.append(elem)
            continue
        st = elem.node.str()
        if st.isdigit():
            dims.append(int(st))
            continue

        if len(dims) > 0:
            generate_digits_op(dims)
            dims = []
        generate_sym_int(elem) 
    if len(dims) > 0:
        generate_digits_op(dims)

    # concat all ops
    op = OP(f"{name}_{suffix}", "ConcatD")
    op.set_dynamic_input("x", len(x_names), x_names)
    op.set_attr_int("N", len(x_names))
    op.set_attr_int("concat_dim", 0)
    ops.append(op.to_node())
    return ops


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def get_ascend_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bool:
        return "BOOL"
    elif dtype == torch.int64:
        return "INT64"
    elif dtype == torch.float32:
        return "FLOAT"
    elif dtype == torch.float16:
        return "FLOAT16"
    elif dtype == torch.int32:
        return "INT32"
    elif dtype == torch.complex64:
        return "COMPLEX64"
    elif dtype == torch.bool:
        return "BOOL"
    else:
        raise RuntimeError("unknow torch data tyep type in get_ascend_dtype!")


def get_np_dtype(dtype):
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.int64:
        return np.int64
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.bool:
        return np.bool_
    elif dtype == torch.float16:
        return np.float16
    raise RuntimeError("unsupported np dtype!")


def get_ascend_dtype_num(dtype: str):
    if dtype == "FLOAT":
        return 0
    elif dtype == "FLOAT16":
        return 1
    elif dtype == "INT32":
        return 3
    elif dtype == "INT64":
        return 9
    elif dtype == "BOOL":
        return 12
    elif dtype == "COMPLEX64":
        return 16
    else:
        raise RuntimeError("unknow torch data tyep type in get_ascend_dtype!")


def get_cpp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "INT64"
    elif dtype == torch.float32:
        return "FLOAT"
    elif dtype == torch.float16:
        return "FLOAT16"
    elif dtype == torch.int32:
        return "INT32"
    else:
        raise RuntimeError("unknow torch data tyep type in get_cpp_dtype!")


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

        global sym_to_inputs
        sym_to_inputs = {}

        super().__init__(graph)

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = name 
        self.input_args.append(self.cur_node)
        fake_tensor = self.cur_node.meta['val']

        format = "NCHW"
        index = -1

        if isinstance(fake_tensor, torch.SymInt):
            dims = [1]
            data_type = "INT32"
            format = "ND"
            sym_to_inputs[fake_tensor.node.str()] = name
        elif symint_in_shape(fake_tensor.shape):
            # deal with dynamic shape -1
            shape = [-1 if isinstance(elem, torch.SymInt) else elem for elem in fake_tensor.shape]
            actual_shape = [elem.node.str() if isinstance(elem, torch.SymInt) else str(elem) for elem in fake_tensor.shape]
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
            format =  self.cur_node.meta['format']
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

        _, args_list = AscendOverrides.gen_args(self.args_dict[name], self.args_dict, self.cur_node, args)
        real_op = process_name(name, target)
        op = getattr(self.override, real_op)(*args_list, **kwargs)
        if isinstance(op, list):
            self.common_nodes.extend(op)
        else:
            self.common_nodes.append(op)


    def get_attr(self, name, target, args, kwargs):
        assert isinstance(target, str)
        attr = self.fetch_attr(target)
        assert(isinstance(attr, torch.Tensor))
        self.args_dict[name] = name 
        op = getattr(self.override, 'get_const_attr')(name, attr)
        self.common_nodes.append(op)


    def call_method(self, name, target, args, kwargs):
        pass

    def output(self, name, target, args, kwargs):
        for arg in args:
            self.output_args.extend(arg)

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

    def gen_import_code(self):
        self.import_code.splice(
            f"""
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
            """
            , strip=True
        )
        return self.import_code.getvalue()

    def process_sym_name(self, st):
        if st.isdigit():
            return st
        elif '+' in st:
            sp = st.split('+')
            assert(len(sp) == 2)
            sp = [elem.strip() for elem in sp]
            return sym_to_inputs[sp[0]] + '+' + sp[1]
        elif '-' in st:
            sp = st.split('-')
            assert(len(sp) == 2)
            sp = [elem.strip() for elem in sp]
            return sym_to_inputs[sp[0]] + '-' + sp[1]
        else:
            return sym_to_inputs[st]

    def gen_call_func(self):
        # TODO check scalar input
        call_body = IndentedBuffer()
        self.args = [self.args_dict[x.name] for x in self.input_args]

        if len(self.dynamic_inputs) > 0:
          args = ['_' if not arg in sym_to_inputs.values() else arg for arg in self.args]
          call_body.writeline(f"({','.join(args)}) = args")
          dim_len = 0
          for shape in self.actual_shape:
              dim_len += len(shape)
          call_body.writeline(f'''max_range = 0''')
          dims = f'''dims = {{'''
          for idx, elem in enumerate(self.actual_shape):
              if len(elem) == 0:
                  continue
              elem = [self.process_sym_name(dim) for dim in elem]
              dims += str(self.dynamic_index[idx]) + ":[" + ','.join(map(str, elem)) + '],'
          dims = dims[:-1] + f'''}}'''
          call_body.writeline(dims)
          unique_elem = []
          for elem in self.actual_shape:
              for dim in elem:
                  st = self.process_sym_name(dim)
                  if not st in unique_elem:
                    unique_elem.append(st)
                    if not st.isdigit():
                        call_body.writeline(f'''if max_range < {st}:''')
                        call_body.writeline(f'''    max_range = {st}''')
        else:
          call_body.writeline(f'''dims = None''')
          call_body.writeline(f'''max_range = 0''')

        call_body.splice(f"""
                             for idx in range(len(args)):
                                 if isinstance(args[idx], int):
                                     args[idx] = torch.tensor(args[idx], device='xpu', dtype=torch.int32)
                         """, strip=True)
        call_body.writeline(f"({','.join(self.args)}) = args")
        call_str = [f'output_tensor = kernel_cpp_0(args, dims, max_range)']
        
        if precision_check and self.aten_graph is not None:
            # 1. export aten graph to disk
            def get_unique_str():
                uuid_str = str(uuid.uuid1())
                return 'm' + str(uuid.uuid3(uuid.NAMESPACE_DNS, uuid_str + str(os.getpid())))
            module_path = get_unique_str().replace('-', '')
            folder_path = "/tmp/dicp_debug/aten_modules/" + module_path
            os.system(f"mkdir -p {folder_path}")
            self.aten_graph.to_folder(folder_path)
          
            # 2. import aten graph
            call_str.append(f'from aten_modules.{module_path} import FxModule')
            call_str.append('aten_call = FxModule()')
            call_str.append('aten_output = aten_call(*args)')

        for i, name in enumerate(self.graph_output_names):
            if not name in self.symint_outputs:
                call_str.append(f'{name} = output_tensor[{i}]')
            else:
                call_str.extend([f'del {name}',
                                 f'{name} = int(output_tensor[{i}])'])
        if precision_check:
            for i, name in enumerate(self.py_output_names):
                if name != 'None' and name not in self.args and name not in self.symint_outputs:
                    call_str.append(f"{name}_cpu = aten_output[{i}]")
                    call_str.append(f"check_tensor({name}, {name}_cpu)")
        call_body.writelines(call_str)

        del_args = [f'del ' + x for x in self.args if x not in self.py_output_names]
        call_body.writelines(del_args)
        call_body.writeline(f"args.clear()")
        call_body.writeline(f"return ({', '.join(self.py_output_names)})")

        call_func = IndentedBuffer()
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
        main_body.writeline(f"print_performance(lambda: call([{', '.join(self.args)}]))")

        main_func = IndentedBuffer()
        main_func.writeline(f"""if __name__ == "__main__":""")
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
                import pdb; pdb.set_trace()   

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
        has_dynamic_shape = False if len(sym_to_inputs) == 0 else True
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
            """
            , strip=True
        )
        compile_graph_code.writeline('async_compile.wait(globals())')
        compile_graph_code.writeline('del async_compile')
        return compile_graph_code.getvalue()

    def generate_code(self):
        return (self.gen_import_code() + self.gen_compile_graph_code()+ self.gen_call_func() + self.gen_main_func())


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
    
    def set_dynamic_input(self, name, num, value):
        assert len(value) == num
        dy_inputs = {
            "name": name,
            "num": num,
            "value": [],
        }
        for i in range(num):
            dy_inputs["value"].append({
                "index": i,
                "value": value[i],
            })
        self.dynamic_inputs.append(dy_inputs)
    
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

    def set_input_with_index(self, name, value, index):
        self.inputs.append({
            "name": name,
            "value": value,
            "index": index,
        })
    
    def set_dynamic_input(self, name, num, value, set_input_with_name = False):
        assert len(value) == num
        dy_inputs = {
            "name": name,
            "num": num,
            "value": [],
        }
        if set_input_with_name == False:
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
    
    def set_dynamic_output(self, name, num):
        self.dynamic_outputs.append({
            "name": name,
            "num": num
        }) 
        
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
    def gen_args(op_var, args_dict, node, args):
        src_code = IndentedBuffer()
        args_str = [op_var]
        if process_name(node.name, node.target) in need_node:
            args_str.append(node)
            
        for i in range(len(args)):
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            else:
                args_str.append(args[i])
        return src_code, args_str

    @staticmethod
    def gen_const_attr(name, val):
        torch_dtype = val.dtype
        shape = list(val.shape)

        assert len(shape) == 0
        assert torch_dtype == torch.float32
        real_val = float(val.numpy().tolist())
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        op = OP(name, "Const")
        op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [real_val], [])
        return op.to_node()

    @staticmethod
    def mul(name, node, x, y):
        (x_node, y_node) = node.args
        ops = []
        dtype = node.meta['val'].dtype
        if not isinstance(y_node, torch.fx.node.Node):
            # y is scalar
            not_support_type = False
            cast_op = None

            if dtype == torch.float16:
                dtype = torch.float32
                cast_op = OP(f'{name}_fp16_cast', "Cast")
                cast_op.set_input('x', f'{name}_broadcast')
                cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))

            try:
                cpp_dtype = get_cpp_dtype(dtype)
                ascend_dtype = get_ascend_dtype(dtype)
            except:
                cpp_dtype = "FLOAT"
                ascend_dtype = "FLOAT"
                not_support_type = True

            mul_op = OP(name, "Mul")
            mul_op.set_input("x1", x)
            y_shape = list(x_node.meta['val'].shape)
            scalar_op = OP(f"{name}_scalar", "Const")
            scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y], [1])
            ops.append(scalar_op.to_node())

            if symint_in_shape(y_shape):
                ops.extend(process_dynamic_shape(y_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
                ops.append(shape_op.to_node())
            broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
            broadcast_op.set_input("x", f"{name}_scalar")
            broadcast_op.set_input("shape", f"{name}_preprocess")
            ops.append(broadcast_op.to_node())
            y_name = f"{name}_broadcast"

            if cast_op is not None:
                ops.append(cast_op.to_node())
                y_name = f'{name}_fp16_cast'

            if not_support_type:
                ascend_dtype = get_ascend_dtype(dtype)
                cast_op1 = OP(f"{name}_scalar_cast", "Cast")
                cast_op1.set_input("x", y_name)
                cast_op1.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
                mul_op.set_input("x2", f"{name}_scalar_cast")
                ops.extend([cast_op1.to_node(), mul_op.to_node()])
                return ops
            else:
                mul_op.set_input("x2", y_name)
                ops.append(mul_op.to_node())
                return ops

        cast_op = None
        y_name = y
        if x_node.meta['val'].dtype == torch.float16:
            cast_op = OP(f'{name}_y_cast', "Cast")
            cast_op.set_input('x', y)
            cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))
            ops.append(cast_op.to_node())
            y_name = f'{name}_y_cast'

        if y_node.meta['val'].dtype != torch.complex64:
            x_name = x
            x_shape = list(x_node.meta['val'].shape)
            y_shape = list(y_node.meta['val'].shape)

            # handling with broadcasting cases
            if np.prod(x_shape) < np.prod(y_shape):
                if symint_in_shape(y_shape):
                    ops.extend(process_dynamic_shape(y_shape, name, 'preprocess_x'))
                else:
                    x_shape_op = OP(f"{name}_preprocess_x", "Const")
                    x_shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
                    ops.append(x_shape_op.to_node())

                broadcast_op = OP(f"{name}_x_broadcast_to", "BroadcastTo")
                broadcast_op.set_input("x", x)
                broadcast_op.set_input("shape", f"{name}_preprocess_x")

                ops.append(broadcast_op.to_node())
                x_name = f"{name}_x_broadcast_to"
            elif np.prod(x_shape) > np.prod(y_shape):
                if symint_in_shape(x_shape):
                    ops.extend(process_dynamic_shape(x_shape, name, 'preprocess_y'))
                else:
                    y_shape_op = OP(f"{name}_preprocess_y", "Const")
                    y_shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", x_shape, [len(x_shape)])
                    ops.append(y_shape_op.to_node())

                broadcast_op = OP(f"{name}_y_broadcast_to", "BroadcastTo")
                broadcast_op.set_input("x", y_name)
                broadcast_op.set_input("shape", f"{name}_preprocess_y")

                ops.append(broadcast_op.to_node())
                y_name = f"{name}_y_broadcast_to"


            x_dtype = x_node.meta['val'].dtype
            y_dtype = y_node.meta['val'].dtype
            if x_dtype != dtype:
                x_name = f"{name}_x1_cast"
                x1 = OP(x_name, "Cast")
                x1.set_input("x", x)
                x1.set_attr_int("dst_type", get_ascend_dtype_num(get_ascend_dtype(dtype)))
                ops.append(x1.to_node())
            if y_dtype != dtype:
                y_name = f"{name}_x2_cast"
                x2 = OP(y_name, "Cast")
                x2.set_input("x", y)
                x2.set_attr_int("dst_type", get_ascend_dtype_num(get_ascend_dtype(dtype)))
                ops.append(x2.to_node())
            op = OP(name, "Mul")
            op.set_input("x1", x_name)
            op.set_input("x2", y_name)
            ops.append(op.to_node())
            return ops

        assert x_node.meta["val"].dtype == torch.complex64
        assert y_node.meta["val"].dtype == torch.complex64
        
        def gen_op(op_type, name, x1, x2):
            op = OP(name, op_type)
            op.set_input("x1", x1)
            op.set_input("x2", x2)
            return op

        # (a + bj)*(c + dj) = (ac - bd)+(ad + bc)j
        a = OP(f"{name}_a", "Identity")
        a.set_input_with_index("x", x, 0)
        b = OP(f"{name}_b", "Identity")
        b.set_input_with_index("x", x, 1)
        c = OP(f"{name}_c", "Identity")
        c.set_input_with_index("x", y, 0)
        d = OP(f"{name}_d", "Identity")
        d.set_input_with_index("x", y, 1)

        ac = gen_op("Mul", f"{name}_ac", f"{name}_a", f"{name}_c")
        bd = gen_op("Mul", f"{name}_bd", f"{name}_b", f"{name}_d")
        ad = gen_op("Mul", f"{name}_ad", f"{name}_a", f"{name}_d")
        bc = gen_op("Mul", f"{name}_bc", f"{name}_b", f"{name}_c")
        ac_bd = gen_op("Sub", f"{name}_ac_bd", f"{name}_ac", f"{name}_bd")
        ad_bc = gen_op("Add", f"{name}_ad_bc", f"{name}_ad", f"{name}_bc")

        id_op = OP(name, "IdentityN")
        id_op.set_dynamic_input("x", 2, [f"{name}_ac_bd", f"{name}_ad_bc"])
        id_op.set_dynamic_output("y", 2)
        ops = [a, b, c, d, ac, bd, ad, bc, ac_bd, ad_bc, id_op]
        return [op.to_node() for op in ops]

    @staticmethod
    def add(name, node, x, y):
        (x_node, y_node) = node.args
        out_dtype = node.meta['val'].dtype
        ops = []
        x_name = x
        y_name = y
        if isinstance(y_node, torch.fx.node.Node):
            x_dtype = x_node.meta['val'].dtype
            y_dtype = y_node.meta['val'].dtype
            if x_dtype != out_dtype:
                ascend_dtype = get_ascend_dtype(out_dtype)
                x_name = f'{name}_x_cast'
                cast_op = OP(f'{name}_x_cast', "Cast")
                cast_op.set_input("x", x)
                cast_op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
                ops.append(cast_op.to_node())
            if y_dtype != out_dtype:
                ascend_dtype = get_ascend_dtype(out_dtype)
                y_name = f'{name}_y_cast'
                cast_op = OP(f'{name}_y_cast', "Cast")
                cast_op.set_input("x", x)
                cast_op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
                ops.append(cast_op.to_node())
        else:
            # y is scalar  
            if out_dtype == torch.float or out_dtype == torch.float16:
                adds_op = OP(name, "Adds")
                adds_op.set_input("x", x_name)
                adds_op.set_attr_float("value", float(y))
                return adds_op.to_node()

            ascend_dtype = get_ascend_dtype(out_dtype)
            cpp_dtype = get_cpp_dtype(out_dtype)
            scalar_op = OP(f'{name}_scalar', "Const")
            scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y], [])
            y_name = f"{name}_scalar"
            ops.append(scalar_op.to_node())
        add_op = OP(name, "Add")
        add_op.set_input("x1", x_name)
        add_op.set_input("x2", y_name)
        ops.append(add_op.to_node())
        return ops

    @staticmethod
    def sub(name, node, x, y, reverse=False):
        (x_node, y_node) = node.args
        x_dtype = node.args[0].meta['val'].dtype
        y_name = y
        ops = []
        if not isinstance(y_node, torch.fx.node.Node):
            cast_op = None
            if x_dtype == torch.float16:
                x_dtype = torch.float32
                cast_op = OP(f'{name}_cast', "Cast")
                cast_op.set_input('x', f'{name}_broadcast')
                cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))

            ascend_dtype = get_ascend_dtype(x_dtype)
            cpp_dtype = get_cpp_dtype(x_dtype)
            y_shape = list(x_node.meta['val'].shape)
            scalar_op = OP(f'{name}_scalar', "Const")
            scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y], [1])
            ops.append(scalar_op.to_node())

            if symint_in_shape(y_shape):
                ops.extend(process_dynamic_shape(y_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
                ops.append(shape_op.to_node())
            broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
            broadcast_op.set_input("x", f"{name}_scalar")
            broadcast_op.set_input("shape", f"{name}_preprocess")
            ops.append(broadcast_op.to_node())
            y_name = f'{name}_broadcast'

            if cast_op is not None:
                ops.append(cast_op.to_node())
                y_name = f'{name}_cast'
            
        sub_op = OP(name, "Sub")
        if not reverse:
            sub_op.set_input("x1", x)
            sub_op.set_input("x2", y_name)
        else:
            sub_op.set_input("x1", y_name)
            sub_op.set_input("x2", x)
        ops.append(sub_op.to_node())
        return ops

    @staticmethod
    def rsub(name, node, x, y):
        return getattr(AscendOverrides, 'sub')(name, node, x, y, True)

    @staticmethod
    def relu(name, x):
        op = OP(name, "Relu")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def silu(name, x):
        silu_op = OP(name, "Swish")
        silu_op.set_input("x", x)
        silu_op.set_attr_float("scale", 1.0)
        return silu_op.to_node()

    @staticmethod
    def transpose(name, node, input, dim0, dim1):
        input_shape = list(node.args[0].meta['val'].shape)
        rank = len(input_shape)
        dim0 = int(dim0)
        dim1 = int(dim1)
        perm = [num for num in range(rank)]
        perm[dim0] = dim1
        perm[dim1] = dim0
        ops = []
        if symint_in_shape(perm):
            ops.extend(process_dynamic_shape(perm, name))
        else:
            const_op = OP(f"{name}_preprocess", "Const")
            const_op.set_attr_tensor("value", "INT32", "INT32", "ND", perm, [rank])
            ops.append(const_op.to_node())
        transpose_op = OP(name, "Transpose")
        transpose_op.set_input("x", input)
        transpose_op.set_input("perm", f"{name}_preprocess")
        ops.append(transpose_op.to_node())
        return ops

    @staticmethod
    def reciprocal(name, x):
        op = OP(name, "Reciprocal")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def sqrt(name, x):
        op = OP(f"{name}_sqrt", "Sqrt")
        op.set_input("x", x)

        # 1. make nan tensor
        zero_op = OP(f"{name}_zero", "ZerosLike")
        zero_op.set_input("x", x)
        nan_op = OP(f"{name}_nan", "Div")
        nan_op.set_input("x1", f"{name}_zero")
        nan_op.set_input("x2", f"{name}_zero")

        # 2. get condition tensor
        cond_op = OP(f"{name}_cond", "Less")
        cond_op.set_input("x1", x)
        cond_op.set_input("x2", f"{name}_zero")

        # 3. post process result
        res_op = OP(name, "Select")
        res_op.set_input("condition", f"{name}_cond")
        res_op.set_input("x1", f"{name}_nan")
        res_op.set_input("x2", f"{name}_sqrt")

        ops = [op, zero_op, cond_op, nan_op, res_op]
        return [n.to_node() for n in ops]

    @staticmethod
    def rsqrt(name, x):
        op = OP(f"{name}_rsqrt", "Rsqrt")
        op.set_input("x", x)

        # 1. make nan tensor
        zero_op = OP(f"{name}_zero", "ZerosLike")
        zero_op.set_input("x", x)
        nan_op = OP(f"{name}_nan", "Div")
        nan_op.set_input("x1", f"{name}_zero")
        nan_op.set_input("x2", f"{name}_zero")

        # 2. get condition tensor
        cond_op = OP(f"{name}_cond", "Less")
        cond_op.set_input("x1", x)
        cond_op.set_input("x2", f"{name}_zero")

        # 3. post process result
        res_op = OP(name, "Select")
        res_op.set_input("condition", f"{name}_cond")
        res_op.set_input("x1", f"{name}_nan")
        res_op.set_input("x2", f"{name}_rsqrt")
        
        ops = [op, zero_op, cond_op, nan_op, res_op]
        return [n.to_node() for n in ops]

    @staticmethod
    def convolution(name, node, input, weight, bias, stride, padding,
                    dilation, transposed, output_padding, groups):
        assert transposed == False
        assert output_padding == [0, 0]
        
        if len(stride) == 2:
            stride = [1, 1, stride[0], stride[1]]
        if len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]
        if len(dilation) == 2:
            dilation = [dilation[0], dilation[0], dilation[1], dilation[1]]
        format = "NCHW" if node.meta['val'].stride()[-1] == 1 else "NHWC"

        op = OP(name, "Conv2D")
        op.set_input("x", input)
        op.set_input("filter", weight)
        op.set_attr_list_int("strides", stride)
        op.set_attr_list_int("pads", padding)
        op.set_attr_list_int("dilations", dilation)
        op.set_attr_int("groups", groups)
        op.set_attr_str("data_format", format)

        if bias != None:
            op.set_input("bias", bias)    
        return op.to_node()

    @staticmethod
    def convert_element_type(name, x, torch_dtype, layout=torch.strided, device='cpu'):
        ascend_dtype = get_ascend_dtype(torch_dtype)
        op = OP(name, "Cast")
        op.set_input("x", x)
        op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
        return op.to_node()

    @staticmethod
    def mean(name, x, dims=[], keepdim=False):
        const_op = OP(f"{name}_axes", "Const")
        const_op.set_attr_tensor("value", "INT32", "INT32", "ND", dims, [] if len(dims) == 0 else [len(dims)])
        mean_op = OP(name, "ReduceMean")
        mean_op.set_input("x", x)
        mean_op.set_input("axes", f"{name}_axes")
        mean_op.set_attr_bool("keep_dims", keepdim)
        return [const_op.to_node(), mean_op.to_node()]

    @staticmethod
    def symsize(name, x, dim):
        dim = [dim] if not isinstance(dim, list) else dim
        op = OP(f"{name}_axis", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", [0], [1])
        op1 = OP(f"{name}_indices", "Const") 
        op1.set_attr_tensor("value", "INT32", "INT32", "ND", dim, [len(dim)])
        op2 = OP(name, "GatherV2")
        op2.set_input("x", x)
        op2.set_input("indices", f"{name}_indices")
        op2.set_input("axis", f"{name}_axis")
        return [op.to_node(), op1.to_node(), op2.to_node()]

    @staticmethod
    def inmul(name, node, x, y):
        (x_node, y_node) = node.args
        assert(not isinstance(y_node, torch.fx.node.Node))
        cpp_dtype = "FLOAT"
        ascend_dtype = "FLOAT"
        const_op = OP(f"{name}_scalar", "Const")
        const_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y], [])
        mul_op = OP(name, "Mul")
        mul_op.set_input("x1", x)
        mul_op.set_input("x2", f"{name}_scalar")
        return [const_op.to_node(), mul_op.to_node()]

    @staticmethod
    def inge(name, node, x, y):
        (x_node, y_node) = node.args
        if not isinstance(y_node, torch.fx.node.Node):
            assert isinstance(y, int)
            cpp_dtype = "INT32"
            ascend_dtype = "INT32"
            const_op = OP(f"{name}_scalar", "Const")
            const_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y], [])
            ge_op = OP(name, "GreaterEqual")
            ge_op.set_input("x1", x)
            ge_op.set_input("x2", f"{name}_scalar")
            return [const_op.to_node(), ge_op.to_node()]
        else:
            ge_op = OP(name, "GreaterEqual")
            ge_op.set_input("x1", x)
            ge_op.set_input("x2", y)
            return ge_op.to_node()

    @staticmethod
    def inadd(name, node, x, y):
        (x_node, y_node) = node.args
        if not isinstance(x_node, torch.fx.node.Node):
            assert isinstance(x, int)
            cpp_dtype = "INT32"
            ascend_dtype = "INT32"
            const_op = OP(f"{name}_scalar", "Const")
            const_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [x], [])
            add_op = OP(name, "AddV2")
            add_op.set_input("x1", f"{name}_scalar")
            add_op.set_input("x2", y)
            return [const_op.to_node(), add_op.to_node()]
        else:
            y_name = y
            ops = []
            if not isinstance(y_node, torch.fx.node.Node):
                assert isinstance(y, int)
                cpp_dtype = "INT32"
                ascend_dtype = "INT32"
                const_op = OP(f"{name}_scalar", "Const")
                const_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y], [])
                ops.append(const_op.to_node())
                y_name = f"{name}_scalar"

            add_op = OP(name, "AddV2")
            add_op.set_input("x1", x)
            add_op.set_input("x2", y_name)
            ops.append(add_op.to_node())
            return ops

    @staticmethod
    def view(name, node, x, size):
        x_node = node.args[0]
        shape = list(node.meta['val'].shape)
        if x_node.meta["val"].dtype == torch.complex64:
            shape.append(1)
        numel = node.meta['val'].numel()
        shape_size = len(shape)

        neg = False
        for i in shape:
            if not isinstance(i, torch.SymInt):
                if i == -1:
                    neg = True
                    break

        if neg:
            prod = 1
            for i in shape:
                if not isinstance(i, torch.SymInt):
                    if i > 0:
                        prod *= i
                else:
                    raise RuntimeError("cannot handle with both negative and symint!")

            real_shape = []
            for i in shape:
                if not isinstance(i, torch.SymInt):
                    if i > 0:
                        real_shape.append(str(i))
                    else:
                        real_shape.append(str(numel / prod))
                else:
                    raise RuntimeError("cannot handle with both negative and symint!")
            shape = real_shape
        ops = []
        if symint_in_shape(shape):
            ops.extend(process_dynamic_shape(shape, name))
        else:
            const_op = OP(f"{name}_preprocess", "Const")    
            const_op.set_attr_tensor("value", "INT32", "INT32", "ND", shape, [shape_size])
            ops.append(const_op.to_node())

        if x_node.meta["val"].dtype == torch.complex64:
            tmp = []
            real = OP(f"{name}_real", "Identity")
            real.set_input_with_index("x", x, 0)
            imag = OP(f"{name}_imag", "Identity")
            imag.set_input_with_index("x", x, 1)
            
            real_reshape = OP(f"{name}_real_reshape", "Reshape")
            real_reshape.set_input("x", f"{name}_real")
            real_reshape.set_input("shape", f"{name}_preprocess")
            imag_reshape = OP(f"{name}_imag_reshape", "Reshape")
            imag_reshape.set_input("x", f"{name}_imag")
            imag_reshape.set_input("shape", f"{name}_preprocess")

            id_op = OP(name, "IdentityN")
            id_op.set_dynamic_input("x", 2, [f"{name}_real_reshape", f"{name}_imag_reshape"])
            id_op.set_dynamic_output("y", 2)
            tmp = [real, imag, real_reshape, imag_reshape, id_op]
            ops.extend([op.to_node() for op in tmp])
        else:
            op = OP(name, "Reshape")
            op.set_input("x", x)
            op.set_input("shape", f"{name}_preprocess")
            ops.append(op.to_node())
        return ops

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
        op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", x_value, x_shape)

        return op.to_node()

    @staticmethod
    def clone(name, x, memory_format=None):
        op = OP(name, "Identity")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def copy(name, dst, src):
        return getattr(AscendOverrides, "clone")(name, src)
      
    @staticmethod
    def _to_copy(name, x, dtype=None, layout=None, device=None):
        if dtype:
            ascend_dtype = get_ascend_dtype(dtype)
            op = OP(name, "Cast")
            op.set_input("x", x)
            op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
            return op.to_node()
        else:
            op = OP(name, "Identity")
            op.set_input("x", x)
            return op.to_node()

    @staticmethod
    def lt(name, node, a, b):
        (x_node, y_node) = node.args
        x_dtype = x_node.meta['val'].dtype
        y_name = b
        ops = []
        if not isinstance(y_node, torch.fx.node.Node):
            cast_op = None
            if x_dtype == torch.float16:
                x_dtype = torch.float32
                cast_op = OP(f'{name}_cast', "Cast")
                cast_op.set_input('x', f'{name}_broadcast')
                cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))

            ascend_dtype = get_ascend_dtype(x_dtype)
            cpp_dtype = get_cpp_dtype(x_dtype)
            y_shape = list(x_node.meta['val'].shape)

            scalar_op = OP(f'{name}_scalar', "Const")
            scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [b], [1])
            ops.append(scalar_op.to_node())
            if symint_in_shape(y_shape):
                ops.extend(process_dynamic_shape(y_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
                ops.append(shape_op.to_node())
            broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
            broadcast_op.set_input("x", f"{name}_scalar")
            broadcast_op.set_input("shape", f"{name}_preprocess")
            ops.append(broadcast_op.to_node())
            y_name = f'{name}_broadcast'

            if cast_op is not None:
                ops.append(cast_op.to_node())
                y_name = f'{name}_cast'

        op = OP(name, "Less")
        op.set_input("x1", a)
        op.set_input("x2", y_name)
        ops.append(op.to_node())
        return ops

    @staticmethod
    def masked_fill(name, node, x, mask, value):
        torch_dtype = node.args[0].meta['val'].dtype
        cast_op = None
        ops = []
        if torch_dtype == torch.float16:
            torch_dtype = torch.float32
            cast_op = OP(f'{name}_cast', "Cast")
            cast_op.set_input('x', f'{name}_broadcast')
            cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))

        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)

        if str(value) == "-inf":
            value = -3.4028235e38

        op = OP(f"{name}_value", "Const")
        op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [value], [1])
        ops.append(op.to_node())
        value_name = f"{name}_value"

        mask_shape = list(node.args[1].meta['val'].shape)
        if symint_in_shape(mask_shape):
            ops.extend(process_dynamic_shape(mask_shape, name))
        else:
            shape_op = OP(f"{name}_preprocess", "Const")
            shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", mask_shape, [len(mask_shape)])
            ops.append(shape_op.to_node())
        broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
        broadcast_op.set_input("x", f"{name}_value")
        broadcast_op.set_input("shape", f"{name}_preprocess")
        ops.append(broadcast_op.to_node())
        value_name = f"{name}_broadcast"

        if cast_op is not None:
            ops.append(cast_op.to_node())
            value_name = f'{name}_cast'

        op1 = OP(name, "MaskedFill")
        op1.set_input("x", x)
        op1.set_input("mask", mask)
        op1.set_input("value", value_name)
        ops.append(op1.to_node())
        return ops

    @staticmethod
    def lift_fresh_copy(name, tensor_constant):
        op = OP(name, "Identity")
        op.set_input("x", tensor_constant)
        return op.to_node()

    @staticmethod
    def unsqueeze(name, x, dim):
        if not isinstance(dim, list):
            dim = [dim]
        op = OP(name, "Unsqueeze")
        op.set_input("x", x)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def squeeze(name, x, dim):
        if not isinstance(dim, list):
            dim = [dim]
        op = OP(name, "Squeeze")
        op.set_input("x", x)
        op.set_attr_list_int("axis", dim)
        
        return op.to_node()

    @staticmethod
    def getitem(name, input, index):
        op = OP(name, "Identity")
        op.set_input_with_index("x", input, index)
        return op.to_node()

    @staticmethod
    def exp(name, x):
        op = OP(name, "Exp")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def embedding(name, weight, indices, padding_idx=-1):
        # TODO! consider situation for padding_idx
        # during training stage
        op = OP(f"{name}_axis", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", [0], [1])
        op1 = OP(f"{name}", "GatherV2")
        op1.set_input("x", weight)
        op1.set_input("indices", indices)
        op1.set_input("axis", f"{name}_axis")

        return [op.to_node(), op1.to_node()]

    @staticmethod
    def sigmoid(name, x):
        op = OP(name, "Sigmoid")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def pow(name, node, x, exp):
        (x_node, exp_node) = node.args
        if isinstance(exp_node, torch.fx.node.Node):
            op = OP(name, "Pow")
            op.set_input("x1", x)
            op.set_input("x2", exp)
            return op.to_node()
        
        # exp is scalar
        dtype = node.meta['val'].dtype
        not_support_type = False
        try:
            cpp_dtype = get_cpp_dtype(dtype)
            ascend_dtype = get_ascend_dtype(dtype)
        except:
            cpp_dtype = "FLOAT"
            ascend_dtype = "FLOAT"
            not_support_type = True
        
        pow_op = OP(name, "Pow")
        pow_op.set_input("x1", x)
        scalar_op = OP(f"{name}_scalar", "Const")
        scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [exp], [])
        if not_support_type:
            ascend_dtype = get_ascend_dtype(dtype)
            cast_op = OP(f"{name}_scalar_cast", "Cast")
            cast_op.set_input("x", f"{name}_scalar")
            cast_op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
            pow_op.set_input("x2", f"{name}_scalar_cast")
            return [scalar_op.to_node(), cast_op.to_node(), pow_op.to_node()]
        else:
            pow_op.set_input("x2", f"{name}_scalar")
            return [scalar_op.to_node(), pow_op.to_node()]

    @staticmethod
    def div(name, node, x, y):
        (x_node, y_node) = node.args
        if isinstance(y_node, torch.fx.node.Node):
            op = OP(name, "DivNoNan")
            op.set_input("x1", x)
            op.set_input("x2", y)
            return op.to_node()
        else:
            ops = []
            assert y_node != 0
            cast_op = None
            dtype = node.meta['val'].dtype
            y_shape = list(x_node.meta['val'].shape)

            if dtype == torch.float16:
                dtype = torch.float32
                cast_op = OP(f'{name}_fp16_cast', "Cast")
                cast_op.set_input('x', f'{name}_broadcast')
                cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))

            cpp_dtype = get_cpp_dtype(dtype)
            ascend_dtype = get_ascend_dtype(dtype)
            scalar_op = OP(f"{name}_scalar", "Const")
            scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [y_node], [])
            ops.append(scalar_op.to_node())

            if symint_in_shape(y_shape):
                ops.extend(process_dynamic_shape(y_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
                ops.append(shape_op.to_node())
            broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
            broadcast_op.set_input("x", f"{name}_scalar")
            broadcast_op.set_input("shape", f"{name}_preprocess")
            ops.append(broadcast_op.to_node())
            y_name = f"{name}_broadcast"

            if cast_op is not None:
                ops.append(cast_op.to_node())
                y_name = f'{name}_fp16_cast'

            div_op = OP(name, "Div")
            div_op.set_input("x1", x)
            div_op.set_input("x2", y_name)
            ops.append(div_op.to_node())

            return ops

    @staticmethod
    def maximum(name, node, a, b):
        (a_node, b_node) = node.args
        ops = []
        b_name = b

        a_shape = list(a_node.meta['val'].shape)
        b_shape = list(b_node.meta['val'].shape)

        if np.prod(b_shape) < np.prod(a_shape):
            if symint_in_shape(a_shape):
                ops.extend(process_dynamic_shape(a_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", a_shape, [len(a_shape)])
                ops.append(shape_op.to_node())
            broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
            broadcast_op.set_input("x", b)
            broadcast_op.set_input("shape", f"{name}_preprocess")
            ops.append(broadcast_op.to_node())
            b_name = f"{name}_broadcast"

        if a_node.meta['val'].dtype == torch.float16:
            cast_op = OP(f'{name}_b_cast', "Cast")
            cast_op.set_input('x', b_name)
            cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))
            ops.append(cast_op.to_node())
            b_name = f'{name}_b_cast'

        op = OP(name, "Maximum")
        op.set_input("x1", a)
        op.set_input("x2", b_name)
        ops.append(op.to_node())

        return ops

    @staticmethod
    def _softmax(name, x, dim=-1, half_to_float=False):
        assert(half_to_float == False)
        op = OP(name, "SoftmaxV2")
        op.set_input("x", x)
        if isinstance(dim, int):
            dim = [dim]
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def sum(name, x, axes=[], keep_dims=False):
        if not isinstance(axes, list):
            axes = [axes]
        op = OP(name, "ReduceSumD")
        op.set_input("x", x)
        op.set_attr_list_int("axes", axes)
        op.set_attr_bool("keep_dims", keep_dims)
        return op.to_node()

    @staticmethod
    def amax(name, x, axes, keep_dims):
        if not isinstance(axes, list):
            axes = [axes]
        op = OP(name, "ReduceMaxD")
        op.set_input("x", x)
        op.set_attr_list_int("axes", axes)
        op.set_attr_bool("keep_dims", keep_dims)
        return op.to_node()

    @staticmethod
    def permute(name, x, order=[0]):
        if order is not None and not isinstance(order, list):
            order = [order]
        op = OP(name, "Permute")
        op.set_input("x", x)
        op.set_attr_list_int("order", order)
        return op.to_node()

    @staticmethod
    def max_pool2d_with_indices(name, x, ksize, strides, padding=[0, 0]):
        assert len(ksize) == 2
        assert len(strides) == 2

        ksize = [1, 1, ksize[0], ksize[1]]
        strides = [1, 1, strides[0], strides[1]]

        ops = []
        fwd_out_op = OP(f"{name}_fwd_out", "MaxPool") 
        if padding != [0, 0]:
            padding = [0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1]]
            pad_const_op = OP(f"{name}_paddings", "Const")
            pad_const_op.set_attr_tensor("value", "INT32", "INT32", "NCHW", padding, [4, 2])
            pad_op = OP(f"{name}_pad", "PadV3")
            pad_op.set_input("x", x)
            pad_op.set_input("paddings", f"{name}_paddings")
            ops.append(pad_const_op.to_node())
            ops.append(pad_op.to_node())
            fwd_out_op.set_input("x", f"{name}_pad")
        else:
            fwd_out_op.set_input("x", x)
        fwd_out_op.set_attr_list_int("ksize", ksize)
        fwd_out_op.set_attr_list_int("strides", strides)
        fwd_out_op.set_attr_str("padding", "VALID")
        fwd_out_op.set_attr_str("data_format", "NCHW")
        ops.append(fwd_out_op.to_node())

        shape_op = OP(f"{name}_shape", "Shape")
        shape_op.set_input("x", f"{name}_fwd_out")
        index_op = OP(f"{name}_indice", "Empty")
        index_op.set_input("shape", f"{name}_shape")
        index_op.set_attr_int("dtype", get_ascend_dtype_num("INT64"))

        id_op = OP(name, "IdentityN")
        id_op.set_dynamic_input("x", 2, [f"{name}_fwd_out", f"{name}_indice"])
        id_op.set_dynamic_output("y", 2)
        
        ops.append(shape_op.to_node())
        ops.append(index_op.to_node())
        ops.append(id_op.to_node())
        return ops

    @staticmethod
    def addmm(name, c, a, b, beta=1.0, alpha=1.0):
        ops = []
        beta_op = OP(f"{name}_beta", "Const")
        beta_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [beta], [])
        alpha_op = OP(f"{name}_alpha", "Const")
        alpha_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [alpha], [])
        ops.append(beta_op.to_node())
        ops.append(alpha_op.to_node())

        c_beta_op = OP(f"{name}_c_beta", "Mul")
        c_beta_op.set_input("x1", c)
        c_beta_op.set_input("x2", f"{name}_beta")
        ops.append(c_beta_op.to_node())

        a_alpha_op = OP(f"{name}_a_alpha", "Mul")
        a_alpha_op.set_input("x1", a)
        a_alpha_op.set_input("x2", f"{name}_alpha")
        ops.append(a_alpha_op.to_node())

        matmul_op = OP(f"{name}_matmul", "MatMul")
        matmul_op.set_input("x1", f"{name}_a_alpha")
        matmul_op.set_input("x2", b)
        ops.append(matmul_op.to_node())

        add_op = OP(name, "Add")
        add_op.set_input("x1", f"{name}_c_beta")
        add_op.set_input("x2", f"{name}_matmul")
        ops.append(add_op.to_node())
        return ops

    @staticmethod
    def var(name, x, axes=[], correction=1, keepdim=True):
        if correction == 1:
            unbiased = True
        elif correction == 0:
            unbiased = False
        else:
            raise RuntimeError("not supported yet!")

        ops = []
        if not isinstance(axes, list):
            axes = [axes]
        axes_op = OP(f"{name}_axes", "Const") 
        axes_op.set_attr_tensor("value", "INT32", "INT32", "ND", axes, [len(axes)] if len(axes) > 0 else [])
        mean_op = OP(f"{name}_mean", "ReduceMean")
        mean_op.set_input("x", x)
        mean_op.set_input("axes", f"{name}_axes")

        input_shape_op = OP(f"{name}_input_shape", "Shape") 
        input_shape_op.set_input("x", x)
        broadcast_op = OP(f"{name}_broadcast_to", "BroadcastTo")
        broadcast_op.set_input("x", f"{name}_mean")
        broadcast_op.set_input("shape", f"{name}_input_shape")

        op = OP(name, "ReduceStdV2Update")
        op.set_input("x", x)
        op.set_input("mean", f"{name}_broadcast_to")
        op.set_attr_list_int("dim", axes)
        op.set_attr_bool("unbiased", unbiased)
        op.set_attr_bool("keepdim", keepdim)

        ops.append(axes_op.to_node())
        ops.append(mean_op.to_node())
        ops.append(input_shape_op.to_node())
        ops.append(broadcast_op.to_node())
        ops.append(op.to_node())
        return ops

    @staticmethod
    def log(name, x):
        op = OP(name, "Log")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def gather(name, node, x, dim, index):
        dim = [dim] if not isinstance(dim, list) else dim
        op = OP(f"{name}_axis", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", dim, [len(dim)])
        op1 = OP(f"{name}", "GatherV2")
        op1.set_input("x", x)
        op1.set_input("indices", index)
        op1.set_input("axis", f"{name}_axis")

        return [op.to_node(), op1.to_node()]

    @staticmethod
    def neg(name, x):
        op = OP(name, "Neg")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def expand(name, node, x, shape):
        x_shape = list(node.args[0].meta['val'].shape)
        y_shape = list(node.meta['val'].shape)
        if x_shape == y_shape:
            op = OP(name, "Identity")
            op.set_input("x", x)
            return op.to_node()

        ops = []
        x_name = x
        if node.args[0].meta['val'].dtype == torch.int64:
            x_name = f"{name}_cast"
            op = OP(x_name, "Cast")
            op.set_input("x", x)
            op.set_attr_int("dst_type", get_ascend_dtype_num("INT32"))
            ops.append(op.to_node())
        
        shape = [dim.meta['val'] if hasattr(dim, 'meta') else dim for dim in shape]
        if isinstance(shape, list) and symint_in_shape(shape):
            ops.extend(process_dynamic_shape(shape, name, "preprocess_shape"))
            op1 = OP(name, "Expand")
            op1.set_input("x", x_name)
            op1.set_input("shape", f"{name}_preprocess_shape")
            ops.append(op1.to_node())
        else:
            op1 = OP(name, "ExpandD")
            op1.set_input("x", x_name)
            op1.set_attr_list_int("shape", shape)
            ops.append(op1.to_node())

        return ops

    @staticmethod
    def zeros_like(name, x, *args):
        # TODO(tangzhiyi): ignore kwargs, need to check this
        op = OP(name, "ZerosLike")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def full(name, node, dims, fill_value, dtype=None, device=None,
             layout=None, pin_memory=False, memory_format=None):
        if len(dims) == 0:
            dims = [1]
        torch_dtype = dtype if dtype else torch.get_default_dtype()
        if hasattr(node.kwargs, 'dtype'):
            torch_dtype = node.kwargs['dtype']
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)

        dims = [dim.meta['val'] if hasattr(dim, 'meta') else dim for dim in dims]
        if hasattr(fill_value, 'meta'):
            fill_value = fill_value.meta['val']

        ops = []
        if symint_in_shape(dims):
            ops.extend(process_dynamic_shape(dims, name))
        else:
            axes_op = OP(f"{name}_preprocess", "Const")
            axes_op.set_attr_tensor("value", "INT32", "INT32", "ND", dims, [len(dims)])
            ops.append(axes_op.to_node())

        val_op = OP(f"{name}_val", "Const")
        val_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [fill_value], [])

        op = OP(f"{name}", "Fill")
        op.set_input("dims", f"{name}_preprocess")
        op.set_input("value", f"{name}_val")
        ops.extend([val_op.to_node(), op.to_node()])

        return ops

    @staticmethod
    def empty(name, size, dtype, layout, device):
        op = OP(f"{name}_shape", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", size, [len(size)])

        dtype = get_ascend_dtype_num(get_ascend_dtype(dtype))
        op1 = OP(f"{name}", "Empty")
        op1.set_input("shape", f"{name}_shape")
        op1.set_attr_int("dtype", dtype)

        return [op.to_node(), op1.to_node()]

    @staticmethod
    def fill(name, node, x, value):
        x_shape = list(node.args[0].meta['val'].shape)
        if len(x_shape) == 0:
            x_shape = [1]
        op = OP(f"{name}_shape", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", x_shape, [len(x_shape)])

        dtype = node.meta['val'].dtype
        cpp_dtype = get_cpp_dtype(dtype)
        ascend_dtype = get_ascend_dtype(dtype)
        assert isinstance(value, int)
        op1 = OP(f"{name}_value", "Const")
        op1.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [value], [])

        op2 = OP(f"{name}", "Fill")
        op2.set_input("dims", f"{name}_shape")
        op2.set_input("value", f"{name}_value")

        return [op.to_node(), op1.to_node(), op2.to_node()]

    @staticmethod
    def repeat_interleave(name, node, repeats, output_size):
        x_shape = list(node.target.repeats.node.meta['val'].shape)
        assert len(x_shape) == 1
        assert x_shape[0] == 1

        op = OP(name, "Const")
        op.set_attr_tensor("value", "INT64", "INT64", "ND", [0], [1])
        '''
        op1 = OP(f"{name}_interleave", "RepeatInterleave")
        op1.set_input("x", f"{name}_value")
        op1.set_input("repeats", repeats)
        op1.set_attr_int("axis", 0)

        op2 = OP(f"{name}_cast", "Cast")
        op2.set_input("x", f"{name}_interleave")
        op2.set_attr_int("dst_type", get_ascend_dtype_num(get_ascend_dtype(dtype)))

        op3 = OP(f"{name}_shape", "Const")
        op3.set_attr_tensor("value", "INT64", "INT64", "ND", x_shape, [len(x_shape)])

        op4 = OP(name, "Reshape")
        op4.set_input("x", f"{name}_cast")
        op4.set_input("shape", f"{name}_shape")
        '''
        # TODO! fix implementation of repeatinterleave
        # Consider situation for repeats > 1

        return op.to_node()

    @staticmethod
    def index_select(name, x, dim, index):
        dim = [dim] if not isinstance(dim, list) else dim
        if isinstance(index, list):
            assert len(index) == 1
            index = str(index[0])
        assert dim[0] == 0

        ops = []
        op = OP(f"{name}_dim", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", dim, [len(dim)])
        ops.append(op.to_node())

        op1 = OP(f"{name}", "GatherV2")
        op1.set_input("x", x)
        op1.set_input("indices", index)
        op1.set_input("axis", f"{name}_dim")
        ops.append(op1.to_node())

        return ops

    @staticmethod
    def index(name, x, index):
        return getattr(AscendOverrides, "index_select")(name, x, 0, index)

    @staticmethod
    def index_arg2_(name, x, index):
        return getattr(AscendOverrides, "index_select")(name, x, 0, index)
    
    @staticmethod
    def index_arg3_(name, x, dim, index):
        return getattr(AscendOverrides, "index_select")(name, x, dim, index)

    @staticmethod
    def ones(name, shape, dtype=torch.int64, device='cpu', pin_memory=False):
        op = OP(f"{name}_shape", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", shape, [len(shape)])

        dtype = get_ascend_dtype_num(get_ascend_dtype(dtype))
        op1 = OP(f"{name}_like", "Empty")
        op1.set_input("shape", f"{name}_shape")
        op1.set_attr_int("dtype", dtype)

        op2 = OP(f"{name}", "OnesLike")
        op2.set_input("x", f"{name}_like")

        return [op.to_node(), op1.to_node(), op2.to_node()]

    @staticmethod
    def new_ones(name, x, shape, dtype=torch.int64, layout=None, device='cpu', pin_memory=False):
        return getattr(AscendOverrides, "ones")(name, shape, dtype, device, pin_memory)

    @staticmethod
    def sort(name, x, dim=-1, descending=False):
        op = OP(name, "Sort")
        op.set_input("x", x)
        op.set_attr_int("axis", dim)
        op.set_attr_bool("descending", descending)

        return op.to_node()

    @staticmethod
    def topk(name, node, x, k, dim=-1, largest=True, sorted=True):
        k_node = node.args[1]
        k_name = k
        ops = []
        if not isinstance(k_node, torch.fx.node.Node):
            k_name = f"{name}_k"
            op = OP(k_name, "Const")
            op.set_attr_tensor("value", "INT32", "INT32", "ND", [k], [])
            ops.append(op.to_node())

        op1 = OP(name, "TopK")
        op1.set_input("x", x)
        op1.set_input("k", k_name)
        op1.set_attr_int("dim", dim)
        op1.set_attr_bool("largest", largest)
        op1.set_attr_bool("sorted", sorted)
        ops.append(op1.to_node())

        return ops

    @staticmethod
    def scatter(name, node, var, dim, index, value):
        assert(len(node.args) > 3)
        index_node = node.args[2]
        value_node = node.args[3]
        index_shape = list(index_node.meta['val'].shape)
        assert isinstance(dim, int)
        
        if isinstance(value_node, torch.fx.node.Node):
            ops = []
            if symint_in_shape(index_shape):
                ops.extend(process_dynamic_shape(index_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "NCHW", index_shape, [len(index_shape)])
                ops.append(shape_op.to_node())

            op = OP(f"{name}_value", "Reshape")
            op.set_input("x", value)
            op.set_input("shape", f"{name}_preprocess")
            ops.append(op.to_node())

            op1 = OP(name, "ScatterElements")
            op1.set_input("data", var)
            op1.set_input("indices", index)
            op1.set_input("updates", f"{name}_value")
            op1.set_attr_int("axis", dim)
            ops.append(op1.to_node())
            return ops
        
        ops = []
        dtype = node.meta['val'].dtype
        ascend_dtype = get_ascend_dtype(dtype)
        cpp_dtype = get_cpp_dtype(dtype)        

        value_op = OP(f"{name}_value")
        value_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [value], [1])
        shape_op = OP(f"{name}_index_shape", "Shape")
        shape_op.set_input("x", index)
        bcast_op = OP(f"{name}_value_bcast", "BroadcastTo")
        bcast_op.set_input("x", f"{name}_value")
        bcast_op.set_input("shape", f"{name}_index_shape")

        op = OP(name, "ScatterElements")
        op.set_input("data", var)
        op.set_input("indices", index)
        op.set_input("updates", f"{name}_value_bcast")
        op.set_attr_int("axis", dim)

        ops.append(value_op.to_node())
        ops.append(shape_op.to_node())
        ops.append(bcast_op.to_node())
        ops.append(op.to_node())
        return ops

    @staticmethod
    def mm(name, node, x, y):
        if node.target.change_input:
            (x, y) = (y, x)
        op = OP(name, "MatMul")
        op.set_input("x1", x)
        op.set_input("x2", y)
        if node.target.trans_a:
            op.set_attr_bool("transpose_x1", True)
        if node.target.trans_b:
            op.set_attr_bool("transpose_x2", True)
        return op.to_node()

    @staticmethod
    def bmm(name, x1, x2, adj_x1=False, adj_x2=False):
        op = OP(name, "BatchMatMul")
        op.set_input("x1", x1)
        if adj_x1:
            op.set_attr_bool("adj_x1", True)
        op.set_input("x2", x2)
        if adj_x2:
            op.set_attr_bool("adj_x2", True)
        return op.to_node()

    @staticmethod
    def convolution_backward(name, grad_output, input, weight, bias_size,
        
                             stride, padding, dilation, transposed, output_padding,
                             groups, grad_input_mask):
        assert transposed == False
        assert output_padding == [0, 0]

        stride = [1, 1, stride[0], stride[1]]
        padding = [padding[0], padding[0], padding[1], padding[1]]
        dilation = [1, 1, dilation[0], dilation[1]]
        data_format = "NCHW"

        ops = []
        if grad_input_mask[0] == True:
            shape_op = OP(f"{name}_input_shape", "Shape")
            shape_op.set_input("x", input)
            bp_op = OP(f"{name}_input", "Conv2DBackpropInput")
            bp_op.set_input("input_size", f"{name}_input_shape")
            bp_op.set_input("filter", weight)
            bp_op.set_input("out_backprop", grad_output)
            bp_op.set_attr_list_int("strides", stride)
            bp_op.set_attr_list_int("pads", padding)
            bp_op.set_attr_list_int("dilations", dilation)
            bp_op.set_attr_int("groups", groups)
            bp_op.set_attr_str("data_format", data_format)
            ops.append(shape_op.to_node())
            ops.append(bp_op.to_node())

        if grad_input_mask[1] == True:
            shape_op = OP(f"{name}_filter_shape", "Shape")
            shape_op.set_input("x", weight)
            bp_op = OP(f"{name}_filter", "Conv2DBackpropFilter")
            bp_op.set_input("x", input)
            bp_op.set_input("filter_size", f"{name}_filter_shape")
            bp_op.set_input("out_backprop", grad_output)
            bp_op.set_attr_list_int("strides", stride)
            bp_op.set_attr_list_int("pads", padding)
            bp_op.set_attr_list_int("dilations", dilation)
            bp_op.set_attr_str("data_format", data_format)
            ops.append(shape_op.to_node())
            ops.append(bp_op.to_node())

        # TODO(tangzhiyi): bias is not supported yet
        assert grad_input_mask[2] == False
        outputs = []
        outputs.append(f"{name}_input" if grad_input_mask[0] else f"{name}_filter")
        outputs.append(f"{name}_filter" if grad_input_mask[1] else f"{name}_input")
        op = OP(name, "IdentityN")
        op.set_dynamic_output("y", 2)
        op.set_dynamic_input("x", 2, outputs)
        ops.append(op.to_node())
        return ops
    
    @staticmethod
    def max_pool2d_with_indices_backward(name, grad_output, x, kernel_size,
                                         stride, padding, dilation, ceil_mode,
                                         indices):
        assert len(kernel_size) == 2 or len(kernel_size) == 1
        assert len(stride) == 2 or len(stride) == 1
        assert len(padding) == 2 or len(padding) == 1
        assert len(dilation) == 2 or len(dilation) == 1
        assert dilation == [1, 1]

        kernel_size = [1, 1, kernel_size[0], kernel_size[1] if len(kernel_size) == 2 else kernel_size[0]]
        stride = [1, 1, stride[0], stride[1] if len(stride) == 2 else stride[0]]
        padding = [1, padding[0], padding[1] if len(padding) == 2 else padding[0], 1]

        ops = []
        if padding != [0, 0]:
            padding = [0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1]]
            op1 = OP(f"{name}_paddings", "Const")
            op1.set_attr_tensor("value", "INT32", "INT32", "NCHW", padding, [8])
            op2 = OP(f"{name}_pad", "PadV3")
            op2.set_input("x", x)
            op2.set_input("paddings", f"{name}_paddings")
            fwd_out = OP(f"{name}_fwd_out", "MaxPool")
            fwd_out.set_input("x", f"{name}_pad")
            fwd_out.set_attr_list_int("ksize", kernel_size)
            fwd_out.set_attr_list_int("strides", stride)
            fwd_out.set_attr_str("padding", "VALID")
            fwd_out.set_attr_str("data_format", "NCHW")
            bwd = OP(f"{name}_bwd", "MaxPoolGrad")
            bwd.set_input("x1", f"{name}_pad")
            bwd.set_input("x2", f"{name}_fwd_out")
            bwd.set_input("grad", grad_output)
            bwd.set_attr_list_int("ksize", kernel_size)
            bwd.set_attr_list_int("strides", stride)
            bwd.set_attr_str("padding", "VALID")
            bwd.set_attr_str("data_format", "NCHW")
            pad_grad = OP(name, "PadV3Grad")
            pad_grad.set_input("x", f"{name}_bwd")
            pad_grad.set_input("paddings", f"{name}_paddings")
            ops.append(op1.to_node())
            ops.append(op2.to_node())
            ops.append(fwd_out.to_node())
            ops.append(bwd.to_node())
            ops.append(pad_grad.to_node())
        else:
            fwd_out = OP(f"{name}_fwd_out", "MaxPool")
            fwd_out.set_input("x", x)
            fwd_out.set_attr_list_int("ksize", kernel_size)
            fwd_out.set_attr_list_int("strides", stride)
            fwd_out.set_attr_str("padding", "VALID")
            fwd_out.set_attr_str("data_format", "NCHW")
            grad = OP(name, "MaxPoolGrad")
            grad.set_input("x1", x)
            grad.set_input("x2", f"{name}_fwd_out")
            grad.set_input("grad", grad_output)
            grad.set_attr_list_int("ksize", kernel_size)
            grad.set_attr_list_int("strides", stride)
            grad.set_attr_str("padding", "VALID")
            grad.set_attr_str("data_format", "NCHW")
            ops.append(fwd_out.to_node())
            ops.append(grad.to_node())
        return ops

    @staticmethod
    def where(name, node, cond, x1, x2):
        # TODO(tangzhiyi): need to process scalars
        (cond_node, x1_node, x2_node) = node.args
        assert isinstance(x1_node, torch.fx.node.Node)
        assert isinstance(x2_node, torch.fx.node.Node)

        ops = []
        shape_op = OP(f"{name}_cond_shape", "Shape")
        shape_op.set_input("x", cond)

        x1_bcast = OP(f"{name}_x1_bcast", "BroadcastTo")        
        x1_bcast.set_input("x", x1)
        x1_bcast.set_input("shape", f"{name}_cond_shape")

        x2_bcast = OP(f"{name}_x2_bcast", "BroadcastTo")
        x2_bcast.set_input("x", x2)
        x2_bcast.set_input("shape", f"{name}_cond_shape")

        op = OP(name, "Select")
        op.set_input("condition", cond)
        op.set_input("x1", f"{name}_x1_bcast")
        op.set_input("x2", f"{name}_x2_bcast")

        ops.append(shape_op.to_node())
        ops.append(x1_bcast.to_node())
        ops.append(x2_bcast.to_node())
        ops.append(op.to_node())
        return ops

    @staticmethod
    def le(name, node, x1, x2):
        (x1_node, x2_node) = node.args
        
        if isinstance(x2_node, torch.fx.node.Node):
            op = OP(name, "LessEqual")
            op.set_input("x1", x1)
            op.set_input("x2", x2)
            return op.to_node()
        
        # TODO(tangzhiyi): get value type, now assume float
        x2_name = f"{name}_x2"
        x2_op = OP(x2_name, "Const")
        x2_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [x2], [])

        cast_op = None
        if x1_node.meta['val'].dtype == torch.float16:
            x2_name = f'{name}_x2_cast'
            cast_op = OP(x2_name, "Cast")
            cast_op.set_input('x', f"{name}_x2")
            cast_op.set_attr_int('dst_type', get_ascend_dtype_num("FLOAT16"))

        op = OP(name, "LessEqual")
        op.set_input("x1", x1)
        op.set_input("x2", x2_name)

        if cast_op is None:
            return [x2_op.to_node(), op.to_node()]
        return [x2_op.to_node(), cast_op.to_node(), op.to_node()]

    @staticmethod
    def scalar_tensor(name, node, val):
        torch_dtype = node.kwargs['dtype']
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        op = OP(name, "Const")
        op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "ND", [val], [])
        return op.to_node()

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
    def arange(name, end, start=0, step=1, device='xpu', pin_memory=False):
        assert isinstance(start, str) or isinstance(start, int)
        assert isinstance(end, str) or isinstance(end, int)
        assert isinstance(step, str) or isinstance(step, int)
        ops = []

        if isinstance(start, str) and start.isdigit():
            start = int(start)
        if isinstance(end, str) and end.isdigit():
            end = int(end)
        if isinstance(step, str) and step.isdigit():
            step = int(step)

        if isinstance(start, int):
            op = OP(f"{name}_start", "Const")
            op.set_attr_tensor("value", "INT32", "INT32", "ND", [int(start)], [])
            start = f"{name}_start"
            ops.append(op.to_node())

        if isinstance(end, int):
            op1 = OP(f"{name}_end", "Const")
            op1.set_attr_tensor("value", "INT32", "INT32", "ND", [int(end)], [])
            end = f"{name}_end"
            ops.append(op1.to_node())

        if isinstance(step, int):
            op2 = OP(f"{name}_step", "Const")
            op2.set_attr_tensor("value", "INT32", "INT32", "ND", [int(step)], [])
            step = f"{name}_step"
            ops.append(op2.to_node())

        op3 = OP(name, "Range")
        op3.set_input("start", start)
        op3.set_input("limit", end)
        op3.set_input("delta", step)
        ops.append(op3.to_node())

        return ops

    @staticmethod
    def eq(name, node, a, b):
        (a_node, b_node) = node.args
        b_name = b
        ops = []
        if not isinstance(b_node, torch.fx.node.Node):
            assert isinstance(b, int)
            b_shape = list(a_node.meta['val'].shape)
            scalar_op = OP(f'{name}_const', "Const")
            scalar_op.set_attr_tensor("value", "INT64", "INT64", "ND", [b], [1])
            ops.append(scalar_op.to_node())

            if symint_in_shape(b_shape):
                ops.extend(process_dynamic_shape(b_shape, name))
            else:
                shape_op = OP(f"{name}_preprocess", "Const")
                shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", b_shape, [len(b_shape)])
                ops.append(shape_op.to_node())
            broadcast_op = OP(f"{name}_broadcast", "BroadcastTo")
            broadcast_op.set_input("x", f"{name}_const")
            broadcast_op.set_input("shape", f"{name}_preprocess")
            ops.append(broadcast_op.to_node())
            b_name = f'{name}_broadcast'

        eq_op = OP(name, "Equal") 
        eq_op.set_input("x1", a)
        eq_op.set_input("x2", b_name)
        ops.append(eq_op.to_node())
        return ops

    @staticmethod
    def cumsum(name, x, dim, dtype=None):
        op = OP(f"{name}_dim", "Const")
        op.set_attr_tensor("value", "INT32", "INT32", "ND", [dim], [1])

        op1 = OP(name, "Cumsum") 
        op1.set_input("x", x)
        op1.set_input("axis", f"{name}_dim")
        return [op.to_node(), op1.to_node()]

    @staticmethod
    def t(name, node, input):
        shape = node.meta['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape.reverse()
        op = OP(name, "Permute") 
        op.set_input("x", input)
        op.set_attr_list_int("order", permute_shape)
        return op.to_node()

    @staticmethod
    def log_softmax(name, x, dim, half_to_float):
        assert half_to_float == False
        dim = [dim] if not isinstance(dim, list) else dim
        op = OP(name, "LogSoftmaxV2")
        op.set_input("logits", x)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def log_softmax_backward(name, grad_output, output, dim, input_dtype):
        dim = [dim] if not isinstance(dim, list) else dim
        op = OP(name, "LogSoftmaxGrad")
        op.set_input("grad", grad_output)
        op.set_input("x", output)
        op.set_attr_list_int("axis", dim)
        return op.to_node()

    @staticmethod
    def nll_loss_forward(name, node, x, target, weight, reduction, ignore_index):
        assert weight == None
        assert ignore_index == -100
        reduction_str = get_reduction_str(reduction)
        csize = [list(node.target.x.node.meta['val'].shape)[1]]

        op1 = OP(f"{name}_target_cast", "Cast")        
        op1.set_input("x", target)
        op1.set_attr_int("dst_type", get_ascend_dtype_num("INT32"))

        op2 = OP(f"{name}_weight", "FillV2D")
        op2.set_attr_float("value", 1.0)
        op2.set_attr_list_int("dims", csize)

        op3 = OP(name, "NLLLoss")
        op3.set_input("x", x)
        op3.set_input("target", f"{name}_target_cast")
        op3.set_input("weight", f"{name}_weight")
        op3.set_attr_str("reduction", reduction_str)
        op3.set_attr_int("ignore_index", ignore_index)

        return [op1.to_node(), op2.to_node(), op3.to_node()]

    @staticmethod
    def native_batch_norm_legit_functional(name, node, x, weight, bias, running_mean,
                                           running_var, train, momentum, eps):
        if train != True:
            raise RuntimeError("not supported yet!")
        x_shape = list(node.target.x.node.meta['val'].shape)
        x_dtype = get_ascend_dtype(node.target.x.node.meta['val'].dtype)

        # 1. get sum and square_sum
        op1 = OP(f"{name}_bn_training_reduce", "BNTrainingReduce")
        # TODO(tangzhiyi): now assume output name is y.
        # TODO(daoxin): potential dynamic shape issue in resnet18
        op1.set_and_update_input("x", x, x_shape, "NCHW", x_dtype)

        # 2. call BNTrainingUpdate
        op2 = OP(f"{name}_bn_training_update", "BNTrainingUpdate")
        op2.set_input("x", x)
        op2.set_input_with_index("sum", f"{name}_bn_training_reduce", 0)
        op2.set_input_with_index("square_sum", f"{name}_bn_training_reduce", 1)
        op2.set_input("scale", weight)
        op2.set_input("offset", bias)
        op2.set_input("mean", running_mean)
        op2.set_input("variance", running_var)
        op2.set_attr_float("epsilon", eps)
        op2.set_attr_float("factor", momentum)
        
        # 3. tie all results: result, saved_mean, saved_invstd
        op3 = OP(name, "IdentityN")
        dynamic_input_names = [
            {"input_name": f"{name}_bn_training_update", "edge_name": "y"},
            {"input_name": f"{name}_bn_training_update", "edge_name": "batch_mean"},
            {"input_name": f"{name}_bn_training_update", "edge_name": "batch_variance"},
            {"input_name": f"{name}_bn_training_update", "edge_name": "mean"},
            {"input_name": f"{name}_bn_training_update", "edge_name": "variance"},
        ]
        op3.set_dynamic_input("x", 5, dynamic_input_names, True)
        op3.set_dynamic_output("y", 5)
        return [op1.to_node(), op2.to_node(), op3.to_node()]
    
    @staticmethod
    def native_batch_norm_backward(name, node, grad_out, x, weight, running_mean, running_var,
            save_mean, save_invstd, train, eps, grad_input_mask):
        x_shape = list(node.target.x.node.meta["val"].shape)
        x_dtype = get_ascend_dtype(node.target.x.node.meta["val"].dtype)

        # get grad_weight and grad_bias
        op1 = OP(f"{name}_update_grad", "BNTrainingUpdateGrad")
        op1.set_and_update_input("grads", grad_out, x_shape, "NCHW", x_dtype, "backprops") 
        op1.set_input("x", x)
        op1.set_input("batch_mean", save_mean)
        op1.set_input("batch_variance", save_invstd)
        op1.set_attr_float("epsilon", eps)

        # get grad_input
        op2 = OP(f"{name}_reduce_grad", "BNTrainingReduceGrad")
        op2.set_input("grads", grad_out)
        op2.set_input("x", x)
        op2.set_input_with_index("diff_scale", f"{name}_update_grad", 0)
        op2.set_input_with_index("diff_offset", f"{name}_update_grad", 1)
        op2.set_input("scale", weight)
        op2.set_input("batch_mean", save_mean)
        op2.set_input("batch_variance", save_invstd)
        op2.set_attr_float("epsilon", eps)

        for mask in grad_input_mask:
            assert mask == True
        op3 = OP(name, "IdentityN")
        dynamic_input_names = [
            {"input_name": f"{name}_reduce_grad", "edge_name": "y"},
            {"input_name": f"{name}_update_grad", "edge_name": "diff_scale"},
            {"input_name": f"{name}_update_grad", "edge_name": "diff_offset"},
        ]
        op3.set_dynamic_input("x", 3, dynamic_input_names, True)
        op3.set_dynamic_output("y", 3)

        return [op1.to_node(), op2.to_node(), op3.to_node()]

    @staticmethod
    def nll_loss_backward(name, node, grad_output, x, target, weight, reduction, ignore_index,
                          total_weight):
        assert weight == None
        assert ignore_index == -100
        reduction_str = get_reduction_str(reduction)
        csize = [list(node.target.x.node.meta['val'].shape)[1]]

        op1 = OP(f"{name}_target_cast", "Cast")
        op1.set_input("x", target)
        op1.set_attr_int("dst_type", get_ascend_dtype_num("INT32"))

        op2 = OP(f"{name}_weight", "FillV2D")
        op2.set_attr_float("value", 1.0)
        op2.set_attr_list_int("dims", csize)

        op3 = OP(name, "NLLLossGrad")
        op3.set_input("x", x)
        op3.set_input("y_grad", grad_output)
        op3.set_input("target", f"{name}_target_cast")
        op3.set_input("weight", f"{name}_weight")
        op3.set_input("total_weight", total_weight)
        op3.set_attr_str("reduction", reduction_str)
        op3.set_attr_int("ignore_index", ignore_index)

        return [op1.to_node(), op2.to_node(), op3.to_node()]

    @staticmethod
    def threshold_backward(name, grad_output, x, threshold):
        if threshold == 0:
            op = OP(name, "ReluGrad")
            op.set_input("gradients", grad_output)
            op.set_input("features", x)
            return op.to_node()
        
        op = OP(name, "ThresholdGradV2D")
        op.set_input("gradients", grad_output)
        op.set_input("features", x)
        op.set_attr_float("threshold", threshold)
        return op.to_node()

    @staticmethod
    def view_as_complex(name, node, x):
        x_shape = list(node.target.x.node.meta['val'].shape)
        x_dtype = node.target.x.node.meta['val'].dtype

        assert x_dtype == torch.float32
        assert x_shape[-1] == 2

        dim = len(x_shape) - 1
        split_op = OP(f"{name}", "SplitD")
        split_op.set_input("x", x)
        split_op.set_attr_int("split_dim", dim)
        split_op.set_attr_int("num_split", 2)
        split_op.set_dynamic_output("y", 2)
        return [split_op.to_node()]      
      
    @staticmethod
    def view_as_real(name, node, x):
        assert node.meta['val'].dtype == torch.float32
        x_shape = list(node.target.x.node.meta['val'].shape)
        dim = len(x_shape)
        
        op1 = OP(f"{name}_getitem_1", "Identity")
        op1.set_input_with_index("x", x, 0)
        op2 = OP(f"{name}_getitem_2", "Identity")
        op2.set_input_with_index("x", x, 1)
        op3 = OP(f"{name}_pack", "Pack")
        op3.set_dynamic_input("x", 2, [f"{name}_getitem_1", f"{name}_getitem_2"])
        op3.set_attr_int("axis", dim)
        op3.set_attr_int("N", 2)
        
        op4 = OP(name, "Squeeze")
        op4.set_input("x", f"{name}_pack")
        op4.set_attr_list_int("x", [-1])
        
        return [op1.to_node(), op2.to_node(), op3.to_node(), op4.to_node()]

    @staticmethod
    def stack(name, x, dim):
        x = [elem.name for elem in x]

        op = OP(f"{name}", "Pack")
        op.set_dynamic_input("x", len(x), x)
        op.set_attr_int("axis", dim)
        op.set_attr_int("N", len(x))

        return op.to_node()

    @staticmethod
    def slice(name, node, x, dim, start, end):
        # TODO(tangzhiyi): miss step parameter
        x_shape = list(node.target.x.node.meta['val'].shape)
        y_shape = list(node.meta['val'].shape)

        dim = int(dim)
        start = int(start)
        start = start if start >= 0 else x_shape[dim] + start
        
        assert dim >= 0 and dim < len(x_shape)
        assert start >=0 and start < x_shape[dim]
        
        offset = [0] * len(x_shape)
        offset[dim] = start
        
        ops = []
        if symint_in_shape(offset):
            ops.extend(process_dynamic_shape(offset, name, "preprocess_offset"))
        else:
            op1 = OP(f"{name}_preprocess_offset", "Const")
            op1.set_attr_tensor("value", "INT32", "INT32", "ND", offset, [len(offset)])
            ops.append(op1.to_node())

        if symint_in_shape(y_shape):
            ops.extend(process_dynamic_shape(y_shape, name, "preprocess_size"))
        else:
            op2 = OP(f"{name}_preprocess_size", "Const")
            op2.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
            ops.append(op2.to_node())
            
        op = OP(name, "Slice")
        op.set_input("x", x)
        op.set_input("offsets", f"{name}_preprocess_offset")
        op.set_input("size", f"{name}_preprocess_size")
        ops.append(op.to_node())
        return ops

    @staticmethod
    def cat(name, node, x, dim=0):
        x_list = [a.name if isinstance(a, torch.fx.node.Node) else a for a in x]
        x_size = len(x_list)
        y_dtype = node.meta['val'].dtype
        
        x_node = node.args[0]
        x_names = []
        ops = []
        for i, v in enumerate(x_node):
            dtype = v.meta['val'].dtype
            if dtype == y_dtype:
                x_names.append(x_list[i])
                continue
            
            # cast to y_dtype
            ascend_dtype = get_ascend_dtype(y_dtype)
            cast_op = OP(f"{name}_cast_{i}", "Cast")
            cast_op.set_input("x", x_list[i])
            cast_op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
            ops.append(cast_op.to_node())
            x_names.append(f"{name}_cast_{i}")

        op = OP(name, "ConcatD")
        op.set_dynamic_input("x", x_size, x_names)
        op.set_attr_int("N", x_size)
        op.set_attr_int("concat_dim", dim)
        ops.append(op.to_node())
        return ops
      
    @staticmethod
    def select(name, node, x, dim, index):
        x_shape = list(node.target.x.node.meta['val'].shape)
        y_shape = list(node.meta['val'].shape)
        dim = int(dim)
        index = int(index)
        
        assert dim >= 0 and dim < len(x_shape)
        start = index if index >= 0 else index + x_shape[dim]
        end = start + 1
        offset = [0] * len(x_shape)
        offset[dim] = start
        size = []
        for i, v in enumerate(x_shape):
            if i != dim:
                size.append(v - offset[i])
            else:
                size.append(end - offset[i])

        ops = []    
        if symint_in_shape(offset):
            ops.extend(process_dynamic_shape(offset, name, "preprocess_offset"))
        else:
            op1 = OP(f"{name}_preprocess_offset", "Const")                        
            op1.set_attr_tensor("value", "INT32", "INT32", "ND", offset, [len(offset)])
            ops.append(op1.to_node())

        if symint_in_shape(size):
            ops.extend(process_dynamic_shape(size, name, "preprocess_size"))
        else:
            op2 = OP(f"{name}_preprocess_size", "Const")
            op2.set_attr_tensor("value", "INT32", "INT32", "ND", size, [len(size)])
            ops.append(op2.to_node())

        op3 = OP(f"{name}_slice", "Slice")
        op3.set_input("x", x)
        op3.set_input("offsets", f"{name}_preprocess_offset")
        op3.set_input("size", f"{name}_preprocess_size")

        if symint_in_shape(y_shape):
            ops.extend(process_dynamic_shape(y_shape, name, "preprocess"))
        else:
            op4 = OP(f"{name}_preprocess", "Const")
            op4.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
            ops.append(op4.to_node())

        op5 = OP(name, "Reshape")
        op5.set_input("x", f"{name}_slice")
        op5.set_input("shape", f"{name}_preprocess")

        ops.append(op3.to_node())        
        ops.append(op5.to_node())
        return ops
      
    @staticmethod
    def rsub(name, node, x, value):
        # this is rsub.scalar
        dtype = node.target.x.node.meta['val'].dtype
        if dtype != torch.float16:
            value_op = OP(f"{name}_value", "Const")
            value_op.set_attr_tensor("value", get_ascend_dtype(dtype), get_cpp_dtype(dtype), "ND", [value], [])
            op = OP(name, "Sub")
            op.set_input("x1", x)
            op.set_input("x2", f"{name}_value")
            return [value_op.to_node(), op.to_node()]
        value_op = OP(f"{name}_value_fp32", "Const")
        value_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [value], [])
        cast_op = OP(f"{name}_value", "Cast")
        cast_op.set_input("x", f"{name}_value_fp32")
        cast_op.set_attr_int("dst_type", get_ascend_dtype_num("FLOAT16"))
        op = OP(name, "Sub")
        op.set_input("x1", x)
        op.set_input("x2", f"{name}_value")
        return [value_op.to_node(), cast_op.to_node(), op.to_node()]
      
    @staticmethod
    def index(name, node, x, index):
        # TODO(tangzhiyi): use Index Op
        assert len(index) == 1
        axis_op = OP(f"{name}_axis", "Const")
        axis_op.set_attr_tensor("value", "INT32", "INT32", "ND", [0], [])
        gather_op = OP(name, "GatherV2")
        gather_op.set_input("x", x)
        gather_op.set_input("indices", index[0].name)
        gather_op.set_input("axis", f"{name}_axis")
        return [axis_op.to_node(), gather_op.to_node()]

    @staticmethod
    def slice_backward(name, node, grad, input_shape, dim, start, end, step):
        start = start if start >= 0 else input_shape[dim] + start
        assert step == 1
        assert dim >= 0 and dim < len(input_shape)
        assert start >=0 and start < input_shape[dim]
        rank = len(input_shape)
        end = end if end <= input_shape[dim] else input_shape[dim]
        end = end if end >=0 else end + input_shape[dim]
        pad = np.zeros((rank, 2), dtype=np.int32)
        for i, v in enumerate(input_shape):
            if i == dim:
                pad[i][0] = start
                pad[i][1] = v - end
        const_op = OP(f"{name}_paddings", "Const")
        const_op.set_attr_tensor("value", "INT32", "INT32", "ND", pad.flatten().tolist(), [rank, 2])
        pad_op = OP(name, "Pad")
        pad_op.set_input("x", grad)
        pad_op.set_input("paddings", f"{name}_paddings")
        return [const_op.to_node(), pad_op.to_node()]
      
    @staticmethod
    def empty_like(name, node, x, dtype=None, device=None, layout=None,
                   pin_memory=False, memory_format=None):
        dtype = get_ascend_dtype(node.target.x.node.meta['val'].dtype)
        shape = list(node.target.x.node.meta['val'].shape)
        
        shape_op = OP(f"{name}_shape", "Const")
        shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", shape, [len(shape)])
        empty_op = OP(name, "Empty")
        empty_op.set_input("shape", f"{name}_shape")
        empty_op.set_attr_int("dtype", get_ascend_dtype_num(dtype))
        return [shape_op.to_node(), empty_op.to_node()]    

    @staticmethod
    def fill_scalar(name, node, x, value):
        fills_op = OP(name, "Fills")
        fills_op.set_input("x", x)
        fills_op.set_attr_float("value", float(value))
        return fills_op.to_node()

    @staticmethod
    def softmax_backward(name, grad_output, output, dim, input_dtype):
        dim = [dim] if not isinstance(dim, list) else dim
        op = OP(name, "SoftmaxGrad")
        op.set_input("grad_softmax", grad_output)
        op.set_input("softmax", output)
        op.set_attr_list_int("axes", dim)
        return op.to_node()

    @staticmethod
    def bernoulli(name, node, x, p, generator=None):
        assert generator is None
        dtype = get_ascend_dtype(node.args[0].meta['val'].dtype)

        shape_op = OP(f"{name}_shape", "Shape")
        shape_op.set_input("x", x)
        prop_op = OP(f"{name}_p", "Const")
        prop_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [float(p)], [])

        # set seed to -1 and offset to 0, so the random number
        # generator is seeded by a random seed.
        seed_op = OP(f"{name}_seed", "Const")
        seed_op.set_attr_tensor("value", "INT64", "INT64", "ND", [-1], [])
        offset_op = OP(f"{name}_offset", "Const")
        offset_op.set_attr_tensor("value", "INT64", "INT64", "ND", [0], [])

        bernoulli_op = OP(name, "StatelessBernoulli")
        bernoulli_op.set_input("shape", f"{name}_shape")
        bernoulli_op.set_input("prob", f"{name}_p")
        bernoulli_op.set_input("seed", f"{name}_seed")
        bernoulli_op.set_input("offset", f"{name}_offset")
        bernoulli_op.set_attr_dtype_str("dtype", dtype)

        ops = [shape_op, prop_op, seed_op, offset_op, bernoulli_op]
        return [op.to_node() for op in ops]

    @staticmethod
    def new_empty_strided(name, node, x, dtype=None, device=None, layout=None,
                   pin_memory=False):
        dtype = get_ascend_dtype(node.target.x.node.meta['val'].dtype)
        shape = list(node.target.x.node.meta['val'].shape)
        
        shape_op = OP(f"{name}_shape", "Const")
        shape_op.set_attr_tensor("value", "INT32", "INT32", "ND", shape, [len(shape)])
        empty_op = OP(name, "Empty")
        empty_op.set_input("shape", f"{name}_shape")
        empty_op.set_attr_int("dtype", get_ascend_dtype_num(dtype))
        return [shape_op.to_node(), empty_op.to_node()]
