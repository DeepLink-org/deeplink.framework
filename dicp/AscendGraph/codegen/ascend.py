import torch
import json

from typing import Any, List
from torch.fx.node import Node
from torch._inductor.utils import IndentedBuffer

graph_id = 0

need_node = ['add', 'mul', 'div', 'view', 'scatter', 'full',
             'where', 'convolution', 'le', 'scalar_tensor',
             't', 'nll_loss_forward', 'native_batch_norm_legit_functional',
             'nll_loss_backward', 'native_batch_norm_backward',
             'view_as_complex', 'view_as_real', 'slice', 'select',
             'pow', 'cat', 'expand', 'transpose', 'inmul']

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


def get_reduction_str(r):
    if r == '0':
        return "none"
    elif r == '1':
        return "mean"
    elif r == '2':
        return "sum"
    else:
        raise RuntimeError("not supported yet!")


def simple_operation(expr, name, token):
    if token == '-':
        opr = 'Sub'
    elif token == '+':
        opr = 'Add'
    else:
        raise RuntimeError("not supported yet!")

    exprs = expr.split(token)
    exprs = [expr.strip() for expr in exprs]
    #!TODO only support tensor minus constant
    assert(len(exprs) == 2)
    try:
        int(exprs[1])
    except ValueError:
        raise RuntimeError("not supported yet!")

    src_code = f"""
                    auto {name}_y_tensor = genTensorWithData<int>({{1}}, FORMAT_NCHW, DT_INT32, {{ {exprs[1]} }});
                    auto {name}_y = op::Const("{name}_y")
                      .set_attr_value({name}_y_tensor);
                    auto {name} = op::{opr}("{name}")
                      .set_input_x1({exprs[0]})
                      .set_input_x2({name}_y);
                    graph.AddOp({name});
                """
    return src_code


def process_shape_str(shape_str, name, suffix=""):
    shape = shape_str.strip('{}').split(',')
    src_code = f""""""
    pattern = []
    count = 0
    for elem in shape:
        elem = elem.strip()
        flag = True
        try:
            int(elem)
        except ValueError:
            flag = False
        if not flag:
            if len(pattern) > 0:
                shape_str = '{' + ','.join(map(str, pattern)) + '}'
                src_code += f"""
                                auto {name}_dim_tensor_{count} = genTensorWithData<int>({{ {len(pattern)} }}, FORMAT_NCHW, DT_INT32, {shape_str});
                                auto {name}_dim_{count} = op::Const("{name}_dim_{count}")
                                  .set_attr_value({name}_dim_tensor_{count});
                            """
                pattern = []
                count += 1

            for key in sym_to_inputs:
                if key in elem:
                    elem = elem.replace(key, sym_to_inputs[key])

            #!TODO deal with more complicated expressions
            if '-' in elem:
                src_code += simple_operation(elem, name+'_dim_'+str(count), '-')
            elif '+' in elem:
                src_code += simple_operation(elem, name+'_dim_'+str(count), '+')
            else:
                src_code += f"""
                                auto& {name}_dim_{count} = {elem};
                             """
            count += 1
        else:
            pattern.append(elem)

    if len(pattern) > 0:
        shape_str = '{' + ','.join(map(str, pattern)) + '}'
        src_code += f"""
                        auto {name}_dim_tensor_{count} = genTensorWithData<int>({{ {len(pattern)} }}, FORMAT_NCHW, DT_INT32, {shape_str});
                        auto {name}_dim_{count} = op::Const("{name}_dim_{count}")
                          .set_attr_value({name}_dim_tensor_{count});
                    """
    else:
        count -= 1
    count += 1

    src_code += f"""
                    auto {name}_preprocess{suffix} = op::ConcatD("{name}_preprocess{suffix}")
                        .create_dynamic_input_x({count})
                """
    for i in range(count):
        src_code += f"""
                        .set_dynamic_input_x({i}, {name}_dim_{i})
                    """
    src_code += f"""
                        .set_attr_concat_dim(0)
                        .set_attr_N({count});
                    graph.AddOp({name}_preprocess{suffix});
                """

    return src_code


def dynamic_shape_str(shape_str):
    shape = shape_str.strip('{}[]').split(',')
    for elem in shape:
        elem = elem.strip()
        flag = True
        try:
            int(elem)
        except ValueError:
            flag = False
        if not flag:
            return True
    return False


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def get_ascend_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "INT64"
    elif dtype == torch.float32:
        return "FLOAT"
    elif dtype == torch.float16:
        return "FLOAT16"
    elif dtype == torch.int32:
        return "INT32"
    elif dtype == torch.complex64:
        return "COMPLEX64"
    else:
        import pdb;pdb.set_trace()
        raise RuntimeError("unknow torch data tyep type in get_ascend_dtype!")
    
def get_ascend_dtype_num(dtype: str):
    if dtype == "FLOAT":
        return 0
    elif dtype == "FLOAT16":
        return 1
    elif dtype == "INT32":
        return 3
    elif dtype == "INT64":
        return 9
    elif dtype == "COMPLEX64":
        return 16
    else:
        import pdb;pdb.set_trace()
        raise RuntimeError("unknow torch data tyep type in get_ascend_dtype!")

def get_cpp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "INT64"
    elif dtype == torch.float32:
        return "FLOAT"
    elif dtype == torch.int32:
        return "INT32"
    else:
        import pdb;pdb.set_trace()
        raise RuntimeError("unknow torch data tyep type in get_cpp_dtype!")

class AscendCodegen(torch.fx.Interpreter):
    def __init__(self, graph):
        self.graph = graph
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

        super().__init__(graph)

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = name 
        self.input_args.append(self.cur_node)
        fake_tensor = self.cur_node.meta['val']

        if isinstance(fake_tensor, torch.SymInt):
            sym_to_inputs.update({fake_tensor.node.str(): self.args_dict[name]})
            tensor_shape = f"{{1}}"
            tensor_format = "FORMAT_ND"
            ascend_dtype = "ge::DataType::DT_INT32"
        elif symint_in_shape(fake_tensor.shape):
            # deal with dynamic shape -1
            shape = [-1 if isinstance(elem, torch.SymInt) else elem for elem in fake_tensor.shape]
            actual_shape = fake_tensor.shape # [elem.node._hint if isinstance(elem, torch.SymInt) else elem for elem in fake_tensor.shape]
            tensor_shape = '{' + ','.join(map(str, shape)) + '}'
            tensor_format = "FORMAT_NCHW"
            ascend_dtype = get_ascend_dtype(fake_tensor.dtype)
            self.dynamic_inputs.append(self.args_dict[name])
            self.dynamic_shape.append(shape)
            self.actual_shape.append(actual_shape)
            self.dynamic_index.append(len(self.input_args_names))
        else:
            tensor_shape = '{' + ','.join(map(str, fake_tensor.shape)) + '}'
            tensor_format = "FORMAT_NCHW"
            ascend_dtype = get_ascend_dtype(fake_tensor.dtype)
            
        # gen data_nodes
        self.data_nodes.append({
            "op_name": self.args_dict[name],
            "op_type": "Data",
            "dims": fake_tensor.shape,
            "format": "NCHW",
            "data_type": get_ascend_dtype(fake_tensor.dtype).upper(),
            "index": -1
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
        for node in self.output_args:
            if isinstance(node, torch.fx.node.Node):
                name = self.args_dict[node.name]
                self.py_output_names.append(name)
                if name in self.graph_output_names:
                    continue
                else:
                    self.graph_output_names.append(name)
                # if node in self.input_args:
                #     self.symint_outputs.append(name)
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
                from dicp.AscendGraph.compile import AsyncCompileAscend
                
                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
            """
            , strip=True
        )
        return self.import_code.getvalue()

    def process_sym_name(self, st):
        if '+' in st:
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
          dims = f'''dims = {{'''
          for idx, elem in enumerate(self.actual_shape):
              if len(elem) == 0:
                  continue
              elem = [self.process_sym_name(dim.node.str()) if isinstance(dim, torch.SymInt) else dim for dim in elem]
              dims += str(self.dynamic_index[idx]) + ":[" + ','.join(map(str, elem)) + '],'
          dims = dims[:-1] + f'''}}'''
        else:
          dims = f'''dims = None'''
        call_body.writeline(dims)

        call_body.splice(f"""
                             for idx in range(len(args)):
                                 if isinstance(args[idx], int):
                                     args[idx] = torch.tensor(args[idx], device='cpu', dtype=torch.int32)
                         """, strip=True)
        call_body.writeline(f"({','.join(self.args)}) = args")
        call_str = [f'output_tensor = kernel_cpp_0(args, dims)']
        for i, name in enumerate(self.graph_output_names):
            if not name in self.symint_outputs:
                call_str.append(f'{name} = output_tensor[{i}]')
            else:
                call_str.extend([f'del {name}',
                                 f'{name} = int(output_tensor[{i}])'])
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
    
    def gen_graph_json(self):
        self.parse_outputs()
        graph = {
            "name": "graph",
            "input_names": self.graph_input_names,
            "output_names": self.graph_output_names,
            "has_dynamic_shape": False,
            "data_nodes": self.data_nodes,
            "common_nodes": self.common_nodes,
        }
        return json.dumps(graph)

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        graph_json = self.gen_graph_json()
        compile_graph_code.splice(
            f"""
                async_compile = AsyncCompileAscend()
                kernel_cpp_0 = async_compile.ascend('''
                {graph_json}
                ''')
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
            
        count = 0
        for i in range(len(args)):
            if isinstance(args[i], Node):
                args_str.append(args_dict[args[i].name])
            # elif isinstance(args[i], bool):
            #     args_str.append(str(args[i]).lower())
            # elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
            #     args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            # elif isinstance(args[i], torch.dtype):
            #     in_shape_size = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
            #     src_code.writeline(f'std::vector<int64_t> {op_var}_shape{count}{in_shape_size};')
            #     args_str.append(f'{op_var}_type{count}')
            #     count += 1
            else:
                args_str.append(args[i])
        return src_code, args_str

    @staticmethod
    def mul(name, node, x, y):
        (_, y_node) = node.args
        if isinstance(y_node, torch.fx.node.Node):
            src_code = f"""
                           auto {name} = op::Mul("{name}")
                             .set_input_x1({x})
                             .set_input_x2({y});
                        """
            op = OP(name, "Mul")
            op.set_input("x1", x)
            op.set_input("x2", y)
            return op.to_node()

        # y is scalar
        dtype = node.meta['val'].dtype
        not_support_type = False
        try:
            cpp_dtype = get_cpp_dtype(dtype)
            ascend_dtype = get_ascend_dtype(dtype)
        except:
            cpp_dtype = "FLOAT"
            ascend_dtype = "FLOAT"
            not_support_type = True
        
        mul_op = OP(name, "Mul")
        mul_op.set_input("x1", x)
        scalar_op = OP(f"{name}_scalar", "Const")
        scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [y], [1])
        if not_support_type:
            ascend_dtype = get_ascend_dtype(dtype)
            cast_op = OP(f"{name}_scalar_cast", "Cast")
            cast_op.set_input("x", f"{name}_scalar")
            cast_op.set_attr_int("dst_type", get_ascend_dtype_num(ascend_dtype))
            mul_op.set_input("x2", f"{name}_scalar_cast")
            return [scalar_op.to_node(), cast_op.to_node(), mul_op.to_node()]
        else:
            mul_op.set_input("x2", f"{name}_scalar")
            return [scalar_op.to_node(), mul_op.to_node()]

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
                src_code += f"""
                            auto {name}_x_cast = op::Cast("{name}_x_cast")
                              .set_input_x({x})
                              .set_attr_dst_type({ascend_dtype});
                          """
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
            ascend_dtype = get_ascend_dtype(out_dtype)
            cpp_dtype = get_cpp_dtype(out_dtype)
            scalar_op = OP(f'{name}_scalar', "Const")
            scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [y], [1])
            y_name = f"{name}_scalar"
            ops.append(scalar_op.to_node())
        add_op = OP(name, "AddV2")
        add_op.set_input("x1", x_name)
        add_op.set_input("x2", y_name)
        ops.append(add_op.to_node())
        return ops

    @staticmethod
    def sub(name, x, y):
        sub_op = OP(name, "Sub")
        sub_op.set_input("x1", x)
        sub_op.set_input("x2", y)
        return sub_op.to_node()

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
        perm_str = '{' + ','.join(map(str, perm)) + '}'
        ops = []
        if dynamic_shape_str(perm_str):
            # TODO(tangzhiyi): dynamic shape process
            src_code = process_shape_str(perm_str, name)
        else:
            const_op = OP(f"{name}_preprocess", "Const")
            const_op.set_attr_tensor("value", "INT32", "INT32", "NCHW", perm, [rank])
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
        op = OP(name, "Sqrt")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def rsqrt(name, x):
        op = OP(name, "Rsqrt")
        op.set_input("x", x)
        return op.to_node()

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
        # TODO(tangzhiyi): dynamic shape process
        src_code = f"""
                       auto {name}_shape = op::Shape("{name}_shape")
                         .set_input_x({x});
                       auto {name}_dim_tensor = genTensorWithData<int>({{1}}, FORMAT_NCHW, DT_INT32, {{ {dim} }});
                       auto {name}_dim = op::Const("{name}_dim")
                         .set_attr_value({name}_dim_tensor);
                       auto {name} = op::Gather("{name}")
                         .set_input_x({x})
                         .set_input_indices({name}_dim);
                       graph.AddOp({name}_shape);
                       graph.AddOp({name}_dim);
                       graph.AddOp({name});
                    """

        return src_code

    @staticmethod
    def inmul(name, node, x, y):
        (x_node, y_node) = node.args
        assert(not isinstance(y_node, torch.fx.node.Node))
        cpp_dtype = "FLOAT"
        ascend_dtype = "FLOAT"
        const_op = OP(f"{name}_scalar", "Const")
        const_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [y], [1])
        mul_op = OP(name, "Mul")
        mul_op.set_input("x1", x)
        mul_op.set_input("x2", f"{name}_scalar")
        return [const_op.to_node(), mul_op.to_node()]

    @staticmethod
    def view(name, node, x, size):
        numel = node.meta['val'].numel()
        shape = list(node.meta['val'].shape)
        shape_size = len(shape)
        if shape.count(-1) > 0:
            prod = 1
            for i in shape:
                if i > 0:
                    prod *= i

            real_shape = []
            for i in shape:
                if i > 0:
                    real_shape.append(str(i))
                else:
                    real_shape.append(str(numel / prod))
            shape = real_shape

        ops = []
        if dynamic_shape_str(str(shape)):
            # TODO(tangzhiyi): dynamic shape process
            #src_code = process_shape_str(shape_str, name)
            pass
        else:
            const_op = OP(f"{name}_preprocess", "Const")    
            const_op.set_attr_tensor("value", "INT32", "INT32", "ND", shape, [shape_size])
            ops.append(const_op.to_node())
        op = OP(name, "Reshape")
        op.set_input("x", x)
        op.set_input("shape", f"{name}_preprocess")
        ops.append(op.to_node())
        return ops

    @staticmethod
    def clone(name, x, memory_format=None):
        op = OP(name, "Identity")
        op.set_input("x", x)
        return op.to_node()
      
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
    def embedding(name, weight, indices):
        op1 = OP(f"{name}_axis", "Const")
        op1.set_attr_tensor("value", "INT32", "INT32", "NCHW", [0], [1])
        op2 = OP(name, "GatherV2")
        op2.set_input("x", weight)
        op2.set_input("indices", indices)
        op2.set_input("axis", f"{name}_axis")
        return [op1.to_node(), op2.to_node()]

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
        scalar_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [exp], [1])
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
        (_, y_node) = node.args
        if isinstance(y_node, torch.fx.node.Node):
            op = OP(name, "DivNoNan")
            op.set_input("x1", x)
            op.set_input("x2", y)
            return op.to_node()
        else:
            div_value = str(1.0 / y_node)
            op = OP(name, "Muls")
            op.set_input("x", x)
            op.set_attr_float("value", div_value)
            return op.to_node()

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
        index_op.set_input("Shape", f"{name}_shape")
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
        beta_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [beta], [1])
        alpha_op = OP(f"{name}_alpha", "Const")
        alpha_op.set_attr_tensor("value", "FLOAT", "FLOAT", "ND", [alpha], [1])
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

        add_op = OP(name, "AddV2")
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
    def gather(name, x, dim, index):
        dim = [dim] if not isinstance(dim, list) else dim
        op1 = OP(f"{name}_dim", "Const")
        op1.set_attr_tensor("value", "INT32", "INT32", "NCHW", dim, [1])
        op2 = OP(name, "GatherD")
        op2.set_input("x", x)
        op2.set_input("dim", f"{name}_dim")
        op2.set_input("index", index)
        op2.set_attr_list_int(dim)
        return [op1.to_node(), op2.to_node()]

    @staticmethod
    def neg(name, x):
        op = OP(name, "Neg")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def expand(name, node, x, shape):
        x_shape = list(node.target.x.node.meta['val'].shape)
        y_shape = list(node.meta['val'].shape)
        if x_shape == y_shape:
            op = OP(name, "Identity")
            op.set_input("x", x)
            return op.to_node()
        
        op = OP(name, "ExpandD")
        op.set_input("x", x)
        op.set_attr_list_int(shape)
        return op.to_node()

    @staticmethod
    def zeros_like(name, x, *args):
        # TODO(tangzhiyi): ignore kwargs, need to check this
        op = OP(name, "ZerosLike")
        op.set_input("x", x)
        return op.to_node()

    @staticmethod
    def full(name, node, dims, fill_value):
        if len(dims) == 0:
            dims = [1]
        torch_dtype = node.kwargs['dtype']
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        
        axes_op = OP(f"{name}_axes", "Const")
        axes_op.set_attr_tensor("value", "INT32", "INT32", "ND", dims, [len(dims)])

        val_op = OP(f"{name}_val", "Const")
        val_op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [fill_value], [1])

        op = OP(name, "Fill")
        op.set_input("dims", f"{name}_axes")
        op.set_input("value", f"{name}_val")

        return [axes_op.to_node(), val_op.to_node(), op.to_node()]

    @staticmethod
    def scatter(name, node, var, dim, index, value):
        assert(len(node.args) > 3)
        value_node = node.args[3]
        dim = [dim] if not isinstance(dim, list) else dim
        
        if isinstance(value_node, torch.fx.node.Node):
            op = OP(name, "ScatterElements")
            op.set_input("data", var)
            op.set_input("updates", value)
            op.set_attr_list_int("axis", dim)
            return op.to_node() 
        
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
        op.set_attr_list_int("axis, dim")

        ops.append(value_op.to_node())
        ops.append(shape_op.to_node())
        ops.append(bcast_op.to_node())
        ops.append(op.to_node())
        return ops

    @staticmethod
    def mm(name, x, y):
        op = OP(name, "MatMul")
        op.set_input("x1", x)
        op.set_input("x2", y)
        return op.to_node()

    @staticmethod
    def bmm(name, x, y):
        op = OP(name, "BatchMatMul")
        op.set_input("x1", x)
        op.set_input("x2", y )
        return op.to_node()

    @staticmethod
    def convolution_backward(name, grad_output, input, weight, bias_size,
        
                             stride, padding, dilation, transposed, output_padding,
                             groups, grad_input_mask):
        # TODO(tangzhiyi): refactor this
        assert transposed == 'false'
        assert output_padding == '{0, 0}'

        src_code = ''
        stride = stride.strip('{}').split(', ')
        padding = padding.strip('{}').split(', ')
        dilation = dilation.strip('{}').split(', ')
        grad_input_mask = grad_input_mask.strip('{}').split(', ')
        new_stride = ['1', '1', stride[0], stride[1]]
        new_padding = [padding[0], padding[0], padding[1], padding[1]]
        new_dilation = ['1', '1', dilation[0], dilation[1]]

        stride_str = '{' + ','.join(new_stride) + '}'
        padding_str = '{' + ','.join(new_padding) + '}'
        dilation_str = '{' + ','.join(new_dilation) + '}'

        # XXX(tangzhiyi): assume data format is NCHW
        data_format = 'NCHW'

        # input
        if grad_input_mask[0] == 'True':
            src_code += f"""
                            auto {name}_input_shape = op::Shape("{name}_input_shape")
                              .set_input_x({input});
                            auto {name}_input = op::Conv2DBackpropInput("{name}_input")
                              .set_input_input_size({name}_input_shape)
                              .set_input_filter({weight})
                              .set_input_out_backprop({grad_output})
                              .set_attr_strides({stride_str})
                              .set_attr_pads({padding_str})
                              .set_attr_dilations({dilation_str})
                              .set_attr_groups({groups})
                              .set_attr_data_format("{data_format}");
                        """

        # weight
        if grad_input_mask[1] == 'True':
            src_code += f"""
                            auto {name}_filter_shape = op::Shape("{name}_filter_shape")
                              .set_input_x({weight});
                            auto {name}_filter = op::Conv2DBackpropFilter("{name}_filter")
                              .set_input_x({input})
                              .set_input_filter_size({name}_filter_shape)
                              .set_input_out_backprop({grad_output})
                              .set_attr_strides({stride_str})
                              .set_attr_pads({padding_str})
                              .set_attr_dilations({dilation_str})
                              .set_attr_groups({groups})
                              .set_attr_data_format("NCHW");
                        """

        # TODO(tangzhiyi): bias is not supported yet
        assert grad_input_mask[2] == 'False'

        only_input = grad_input_mask[0] == 'True' and grad_input_mask[1] == 'False'
        only_weight = grad_input_mask[0] == 'False' and grad_input_mask[1] == 'True'
        both_input_weight = grad_input_mask[0] == 'True' and grad_input_mask[1] == 'True'

        if only_input:
            src_code += f"""
                            auto {name} = op::IdentityN("{name}")
                              .create_dynamic_input_x(2)
                              .set_dynamic_input_x(0, {name}_input)
                              .set_dynamic_input_x(1, {name}_input)
                              .create_dynamic_output_y(2);
                        """
        elif only_weight:
            src_code += f"""
                            auto {name} = op::IdentityN("{name}")
                              .create_dynamic_input_x(2)
                              .set_dynamic_input_x(0, {name}_filter)
                              .set_dynamic_input_x(1, {name}_filter)
                              .create_dynamic_output_y(2);
                        """
        elif both_input_weight:
            src_code += f"""
                            auto {name} = op::IdentityN("{name}")
                              .create_dynamic_input_x(2)
                              .set_dynamic_input_x(0, {name}_input)
                              .set_dynamic_input_x(1, {name}_filter)
                              .create_dynamic_output_y(2);
                        """
        else:
            raise RuntimeError('not supported!')

        return src_code

    @staticmethod
    def max_pool2d_with_indices_backward(name, grad_output, x, kernel_size,
                                         stride, padding, dilation, ceil_mode,
                                         indices):
        # TODO(tangzhiyi): refactor this
        assert len(kernel_size.split(',')) == 2 or len(kernel_size.split(',')) == 1
        assert len(stride.split(',')) == 2 or len(stride.split(',')) == 1
        assert len(padding.split(',')) == 2 or len(padding.split(',')) == 1
        assert len(dilation.split(',')) == 2 or len(dilation.split(',')) == 1

        kernel_size = kernel_size.strip('{}').split(', ')
        stride = stride.strip('{}').split(', ')
        padding = padding.strip('{}').split(', ')
        dilation = dilation.strip('{}').split(', ')

        new_kernel_size = ['1', '1', kernel_size[0], kernel_size[1]]
        new_stride = ['1', '1', stride[0], stride[1]]
        new_padding = ['1', padding[0], padding[1], '1']

        kernel_size_str = '{' + ','.join(new_kernel_size) + '}'
        stride_str = '{' + ','.join(new_stride) + '}'
        padding_str = '{' + ','.join(new_padding) + '}'

        assert dilation == ['1', '1']

        if padding != ['0', '0']:
            padding0 = padding[0]
            padding1 = padding[1]
            padding_str = f'0, 0, 0, 0, {padding0}, {padding0}, {padding1}, {padding1}'
            src_code = f"""
                           auto {name}_pad_tensor = genTensorWithData<int>({{}}, FORMAT_NCHW, DT_INT32, {{ {padding_str} }});
                           auto {name}_paddings = op::Const("{name}_paddings")
                             .set_attr_value({name}_pad_tensor);
                           auto {name}_pad = op::PadV3("{name}_pad")
                             .set_input_x({x})
                             .set_input_paddings({name}_paddings);
                           auto {name}_fwd_out = op::MaxPool("{name}_fwd_out")
                             .set_input_x({name}_pad)
                             .set_attr_ksize({kernel_size_str})
                             .set_attr_strides({stride_str})
                             .set_attr_padding("VALID")
                             .set_attr_data_format("NCHW");
                   
                           auto {name}_bwd = op::MaxPoolGrad("{name}_bwd")
                             .set_input_x1({name}_pad)
                             .set_input_x2({name}_fwd_out)
                             .set_input_grad({grad_output})
                             .set_attr_ksize({kernel_size_str})
                             .set_attr_strides({stride_str})
                             .set_attr_padding("VALID")
                             .set_attr_data_format("NCHW");
                           auto {name} = op::PadV3Grad("{name}")
                             .set_input_x({name}_bwd)
                             .set_input_paddings({name}_paddings);
                        """
        else:
            src_code = f"""
                           auto {name}_fwd_out = op::MaxPool("{name}_fwd_out")
                             .set_input_x({x})
                             .set_attr_ksize({kernel_size_str})
                             .set_attr_strides({stride_str})
                             .set_attr_padding("VALID")
                             .set_attr_data_format("NCHW");
                           auto {name} = op::MaxPoolGrad("{name}")
                             .set_input_x1({x})
                             .set_input_x2({name}_fwd_out)
                             .set_input_grad({grad_output})
                             .set_attr_ksize({kernel_size_str})
                             .set_attr_strides({stride_str})
                             .set_attr_padding("VALID")
                             .set_attr_data_format("NCHW");
                        """

        return src_code

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
        x2_op = OP(f"{name}_x2", "Const")
        x2_op.set_attr_tensor("value", "FLOAT", "FLOAT", "NCHW", [x2], [1])
        op = OP(name, "LessEqual")
        op.set_input("x1", x1)
        op.set_input("x2", f"{name}_x2")
        return [x2_op.to_node(), op.to_node()]

    @staticmethod
    def scalar_tensor(name, node, val):
        torch_dtype = node.kwargs['dtype']
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        op = OP(name, "Const")
        op.set_attr_tensor("value", ascend_dtype, cpp_dtype, "NCHW", [val], [1])
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
        assert weight == 'None'
        assert ignore_index == '-100'
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
    def native_batch_norm_legit_functional(name, x, weight, bias, running_mean,
                                           running_var, train, momentum, eps, node):
        # TODO(tangzhiyi): refactor this
        if train != 'true':
            raise RuntimeError('not supported yet!')
        x_shape = list(node.target.x.node.meta['val'].shape)
        x_shape_str = '{' + ','.join(map(str, x_shape)) + '}'
        x_dtype = get_ascend_dtype(node.target.x.node.meta['val'].dtype)
        src_code = f"""// 0. change x format to NCHW
                       TensorDesc {name}_x_desc(ge::Shape({x_shape_str}), FORMAT_NCHW, {x_dtype});
                       // TODO(tangzhiyi): now assume output name is y.
                       {x}.update_output_desc_y({name}_x_desc);

                       // 1. get sum and square_sum
                       auto {name}_bn_training_reduce = op::BNTrainingReduce("{name}_bn_training_reduce")
                         .set_input_x({x});
                       
                       // 2. call BNTrainingUpdate
                       auto {name}_bn_training_update = op::BNTrainingUpdate("{name}_bn_training_update")
                         .set_input_x({x}, FORMAT_NCHW)
                         .set_input_sum({name}_bn_training_reduce, 0)
                         .set_input_square_sum({name}_bn_training_reduce, 1)
                         .set_input_scale({weight}, FORMAT_NCHW)
                         .set_input_offset({bias}, FORMAT_NCHW)
                         .set_input_mean({running_mean})
                         .set_input_variance({running_var})
                         .set_attr_epsilon({eps})
                         .set_attr_factor({momentum});
                           
                       // 3. tie all results: result, saved_mean, saved_invstd
                       auto {name} = op::IdentityN("{name}")
                         .create_dynamic_input_x(5)
                         .set_dynamic_input_x(0, {name}_bn_training_update, "y") 
                         .set_dynamic_input_x(1, {name}_bn_training_update, "batch_mean")
                         .set_dynamic_input_x(2, {name}_bn_training_update, "batch_variance")
                         .set_dynamic_input_x(3, {name}_bn_training_update, "mean")
                         .set_dynamic_input_x(4, {name}_bn_training_update, "variance")
                         .create_dynamic_output_y(5);
                       """
        return src_code

    @staticmethod
    def native_batch_norm_backward(name, node, grad_out, x, weight, running_mean, running_var,
            save_mean, save_invstd, train, eps, grad_input_mask):
        # TODO(tangzhiyi): refactor this
        x_shape = list(node.target.x.node.meta['val'].shape)
        x_shape_str = '{' + ','.join(map(str, x_shape)) + '}'
        x_dtype = get_ascend_dtype(node.target.x.node.meta['val'].dtype)
        src_code = f"""// 0. change x format to NCHW
                       TensorDesc {name}_x_desc(ge::Shape({x_shape_str}), FORMAT_NCHW, {x_dtype});
                       // TODO(tangzhiyi): now assume output name is y.
                       {x}.update_output_desc_y({name}_x_desc);
                       //{grad_out}.update_output_desc_y({name}_x_desc);
                       
                       // get grad_weight and grad_bias
                       auto {name}_update_grad = op::BNTrainingUpdateGrad("{name}_update_grad")
                         .set_input_grads({grad_out}) 
                         .set_input_x({x})
                         .set_input_batch_mean({save_mean})
                         .set_input_batch_variance({save_invstd})
                         .set_attr_epsilon({eps});

                       // get grad_input
                       auto {name}_reduce_grad = op::BNTrainingReduceGrad("{name}_reduce_grad")
                         .set_input_grads({grad_out})
                         .set_input_x({x})
                         .set_input_diff_scale({name}_update_grad, 0)
                         .set_input_diff_offset({name}_update_grad, 1)
                         .set_input_scale({weight})
                         .set_input_batch_mean({save_mean})
                         .set_input_batch_variance({save_invstd})
                         .set_attr_epsilon({eps});
                       """
                       
        mask = list(map(bool, grad_input_mask.strip('{}').split(',')))
        if mask[0] == True and mask[1] == True and mask[2] == True:
            src_code += f"""
                       auto {name} = op::IdentityN("{name}")
                         .create_dynamic_input_x(3)
                         .set_dynamic_input_x(0, {name}_reduce_grad, "y")
                         .set_dynamic_input_x(1, {name}_update_grad, "diff_scale")
                         .set_dynamic_input_x(2, {name}_update_grad, "diff_offset")
                         .create_dynamic_output_y(3);
                       """
        else:
            raise RuntimeError("not supported yet!")
        return src_code

    @staticmethod
    def nll_loss_backward(name, node, grad_output, x, target, weight, reduction, ignore_index,
                          total_weight):
        assert weight == 'None'
        assert ignore_index == '-100'
        reduction_str = get_reduction_str(reduction)
        csize = [list(node.target.x.node.meta['val'].shape)[1]]

        op1 = OP(f"{name}_target_cast", "Cast")
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
        y_shape = list(node.meta['val'].shape)
        assert x_shape[-1] == 2

        dim = len(x_shape) - 1
        shape_size = len(y_shape)
        shape_str = '{' + ','.join(map(str, y_shape)) + '}'
        output_dtype = 'DT_COMPLEX64' if x_dtype == torch.float32 else 'DT_COMPLEX128'

        ops = []
        split_op = OP(f"{name}_split", "SplitD")
        split_op.set_input("x", x)
        split_op.set_attr_int("split_dim", dim)
        split_op.set_attr_int("num_split", 2)
        split_op.set_dynamic_output("y", 2)
        
        complex_op = OP(f"{name}_complex", "Complex")
        complex_op.set_input_with_index("real", f"{name}_split", 0)
        complex_op.set_input_with_index("imag", f"{name}_split", 1)
        
        ops.append(split_op.to_node())
        ops.append(complex_op.to_node())

        if dynamic_shape_str(str(y_shape)):
            # TODO(tangzhiyi): dynamic shape process
            src_code += process_shape_str(shape_str, name)
        else:
            const_op = OP(f"{name}_preprocess", "Const") 
            const_op.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])
            ops.append(const_op.to_node())
            
        op = OP(name, "Reshape")
        op.set_input("x", f"{name}_complex")
        op.set_input("shape", f"{name}_preprocess")
        ops.append(op.to_node())
        return ops
      
    @staticmethod
    def view_as_real(name, node, x):
        assert node.meta['val'].dtype == torch.float32
        x_shape = list(node.target.x.node.meta['val'].shape)
        dim = len(x_shape) - 1
        
        op1 = OP(f"{name}_real", "Real")
        op1.set_input("input", x)
        op2 = OP(f"{name}_imag", "Imag")
        op2.set_input("input", x)
        op3 = OP(name, "Pack")
        op3.set_dynamic_input("x", 2, [f"{name}_real", f"{name}_imag"])
        op3.set_attr_int("axis", dim)
        op3.set_attr_int("N", 2)

        return [op1.to_node(), op2.to_node(), op3.to_node()]

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
        if dynamic_shape_str(str(offset)):
            # TODO(tangzhiyi): dynamic shape process
            pass
        else:
            op1 = OP(f"{name}_preprocess_offset", "Const")
            op1.set_attr_tensor("value", "INT32", "INT32", "ND", offset, [len(x_shape)])
            ops.append(op1.to_node())

        if dynamic_shape_str(str(y_shape)):
            # TODO(tangzhiyi): dynamic shape process
            pass
        else:
            op2 = OP(f"{name}_preprocess_size", "Const")
            op2.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(x_shape)])
            ops.append(op2.to_node())
            
        op = OP(name, "Slice")
        op.set_input("x", x)
        op.set_input("offsets", f"{name}_preprocess_offset")
        op.set_input("size", f"{name}_preprocess_size")
        ops.append(op.to_node())
        return ops

    @staticmethod
    def cat(name, node, x, dim=0):
        #x_list = x.strip('{}').split(', ')
        x_list = [a.name for a in x]
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
        op.set_attr_int("concat_dim", dim)
        op.set_attr_int("N", x_size)
        ops.append(op.to_node())
        return ops
      
    @staticmethod
    def select(name, node, x, dim, index):
        x_shape = list(node.target.x.node.meta['val'].shape)
        y_shape = list(node.meta['val'].shape)
        dim = int(dim)
        index = int(index)
        
        assert dim >= 0 and dim < len(x_shape)
        start = index + x_shape[dim]
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
        if dynamic_shape_str(str(offset)):
            # TODO(tangzhiyi): dynamic shape process
            pass
        else:
            op1 = OP(f"{name}_preprocess_offset", "Const")                        
            op1.set_attr_tensor("value", "INT32", "INT32", "ND", offset, [len(x_shape)])
            ops.append(op1.to_node())

        if dynamic_shape_str(str(size)):
            # TODO(tangzhiyi): dynamic shape process
            pass
        else:
            op2 = OP(f"{name}_preprocess_size", "Const")
            op2.set_attr_tensor("value", "INT32", "INT32", "ND", size, [len(x_shape)])
            ops.append(op2.to_node())

        op3 = OP(f"{name}_slice", "Slice")
        op3.set_input("x", x)
        op3.set_input("offsets", f"{name}_preprocess_offset")
        op3.set_input("size", f"{name}_preprocess_size")

        op4 = OP(f"{name}_reshape", "Const")
        op4.set_attr_tensor("value", "INT32", "INT32", "ND", y_shape, [len(y_shape)])

        op5 = OP(name, "Reshape")
        op5.set_input("x", f"{name}_slice")
        op5.set_input("shape", f"{name}_reshape")

        ops.append(op3.to_node())
        ops.append(op4.to_node())
        ops.append(op5.to_node())
        return ops
