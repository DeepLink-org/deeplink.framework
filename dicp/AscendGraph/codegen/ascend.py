import torch

from typing import Any
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
    shape = shape_str.strip('{}').split(',')
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


class AscendCodegen(torch.fx.Interpreter):
    def __init__(self, graph):
        self.graph = graph
        self.override = AscendOverrides

        self.import_code = IndentedBuffer()
        self.build_graph_code = IndentedBuffer(initial_indent=1)

        self.graph_id = str(get_graph_id())
        self.args_dict = {}
        self.input_args = []
        self.input_args_names = []
        self.output_args = []

        self.dynamic_inputs = []
        self.dynamic_shape = []
        self.actual_shape = []
        self.dynamic_index = []

        self.symint_outputs = []

        sym_to_inputs = {}

        super().__init__(graph)

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = name 
        self.input_args.append(self.cur_node)
        fake_tensor = self.cur_node.meta['val']

        try:
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
        except:
            import pdb;pdb.set_trace()

        self.build_graph_code.writeline(f'''auto {self.args_dict[name]} = genInput("{self.args_dict[name]}", {tensor_shape}, {tensor_format}, {ascend_dtype}, {len(self.input_args_names)});''')
        self.input_args_names.append(self.args_dict[name])

    def call_function(self, name, target, args, kwargs):
        if name not in self.args_dict.keys():
            self.args_dict[name] = name

        arg_code, args_list = AscendOverrides.gen_args(self.args_dict[name], self.args_dict, self.cur_node, args)

        real_op = process_name(name, target)
        op_code = getattr(self.override, real_op)(*args_list, **kwargs)
        self.build_graph_code.splice(arg_code)
        self.build_graph_code.splice(op_code, strip=True)

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

    def gen_build_graph_code(self):
        graph_code = IndentedBuffer()
        graph_code.splice(self.build_graph_code, strip=True)
        output_str = []
        self.output_names = []
        self.graph_output_names = []
        unique_output_names = []
        for node in self.output_args:
            if isinstance(node, torch.fx.node.Node):
                name = self.args_dict[node.name]
                self.output_names.append(name)
                if name in unique_output_names:
                    continue
                else:
                    unique_output_names.append(name)
                    self.graph_output_names.append(name)

                #!TODO any more accurate method 
                if node in self.input_args:
                    self.symint_outputs.append(name)
            else:
                self.output_names.append(str(node))
        input_str = 'std::vector<Operator> graph_inputs {' + ','.join(self.input_args_names) + '};'
        output_str = 'std::vector<Operator> graph_outputs {' + ','.join(unique_output_names) + '};'
        graph_code.writeline(input_str)
        graph_code.writeline(output_str)
        graph_code.writeline(f'graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);');
        graph_code.writeline(f'return 0;')

        return graph_code

    def get_kernel_header(self):
        return f"""
                #include "graph_utils.h"
                #include <iostream>
                #include <fstream>
                #include <string.h>
                #include <stdint.h>
                #include <memory>
                #include <numeric>
                #include <functional>
               """

    def gen_import_code(self):
        self.import_code.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
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
        call_body.writeline(f"inputs_data = list(map(lambda x: x.data_ptr(), args))")
        call_body.writeline(f"args.clear()")

        call_str = [f'output_np = kernel_cpp_0(inputs_data, dims)']
        for i, name in enumerate(self.graph_output_names):
            if not name in self.symint_outputs:
                call_str.append(f'{name} = torch.from_numpy(output_np[{i}])')
            else:
                call_str.extend([f'del {name}',
                                 f'{name} = int(output_np[{i}])'])
        call_body.writelines(call_str)
        del_args = [f'del ' + x for x in self.args if x not in self.output_names]
        call_body.writelines(del_args)
        call_body.writeline(f"return ({', '.join(self.output_names)})")

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
                # shape = tuple(val.size())
                #if name in self.batch_input_convert.keys():
                #    shape.insert(0, 1)
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

    def gen_compile_func_code(self):
        compile_func_body = IndentedBuffer()
        with compile_func_body.indent():
            compile_func_body.splice(
                f"""
                    std::string graph_name = "BuildGraph" + graph_id;
                    Graph graph(graph_name.c_str());
                    Status ret = genGraph(graph);
                    if (ret != SUCCESS) {{
                        std::cout << "Generate simple graph failed."<<std::endl;
                        return FAILED;
                    }}
                    std::cout<<"Generate simple graph success."<<std::endl;
                """
                , strip=True
            )

            compile_func_body.writeline(f'''std::map<AscendString, AscendString> options;''')
            if len(self.dynamic_inputs) > 0:
              compile_func_body.writeline(f'''options.insert({{''')
              compile_func_body.writeline(f'''{{ge::ir_option::INPUT_FORMAT, "NCHW"}},''')
              code_str = f'''{{ge::ir_option::INPUT_SHAPE, "'''

              for idx, name in enumerate(self.dynamic_inputs):
                  code_str += name + ':'
                  code_str += ','.join(map(str, self.dynamic_shape[idx])) + ';'
              code_str = code_str[:-1] + f'''"}},\n'''
              code_str += f'''}});\n'''
              compile_func_body.writeline(code_str)

            compile_func_body.splice(
                f"""
                    AclgraphBuilder builder;
                    builder.saveGraph(graph_path, graph, options);
                    std::cout << "graph path: " << graph_path << std::endl;
                    return SUCCESS;
                """
                , strip=True
            )

        compile_func = IndentedBuffer()
        compile_func.writeline(f'extern "C" int compile(char* graph_path) {{')
        with compile_func.indent():
           compile_func.splice(compile_func_body)

        compile_func.splice(f"""
                             }}
                             ''')
                             """, strip=True)

        return compile_func

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        compile_graph_code.splice(
            f"""
                async_compile = AsyncCompileAscend()
                kernel_cpp_0 = async_compile.ascend('''
            """
            , strip=True
        )

        compile_graph_code.splice(self.get_kernel_header(), strip=True)
        src_code = f'uint32_t graph_id = {self.graph_id};\n'
        src_code += f'int32_t genGraph(Graph& graph) {{\n'
        compile_graph_code.splice(src_code)
        
        with compile_graph_code.indent():
            compile_graph_code.splice(self.gen_build_graph_code())

        compile_graph_code.writeline(f'}}')
        compile_graph_code.splice(self.gen_compile_func_code())

        compile_graph_code.writeline('async_compile.wait(globals())')
        compile_graph_code.writeline('del async_compile')

        return compile_graph_code.getvalue()

    def generate_code(self):
        return (self.gen_import_code() + self.gen_compile_graph_code()+ self.gen_call_func() + self.gen_main_func())


def get_ascend_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "ge::DataType::DT_INT64"
    elif dtype == torch.float32:
        return "ge::DataType::DT_FLOAT"
    elif dtype == torch.float16:
        return "ge::DataType::DT_FLOAT16"
    elif dtype == torch.int32:
        return "ge::DataType::DT_INT32"
    elif dtype == torch.complex64:
        return "ge::DataType::DT_COMPLEX64"
    else:
        import pdb;pdb.set_trace()
        raise RuntimeError("unknow torch data tyep type in get_ascend_dtype!")


def get_cpp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "int64_t"
    elif dtype == torch.float32:
        return "float"
    else:
        raise RuntimeError("unknow torch data tyep type in get_cpp_dtype!")


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
            elif isinstance(args[i], bool):
                args_str.append(str(args[i]).lower())
            elif isinstance(args[i], torch.fx.immutable_collections.immutable_list):
                args_str.append(str(args[i]).replace('[', '{').replace(']', '}'))
            elif isinstance(args[i], torch.dtype):
                in_shape_size = '{' + str(node.meta['val'].shape).split('[')[-1].split(']')[0] + '}'
                src_code.writeline(f'std::vector<int64_t> {op_var}_shape{count}{in_shape_size};')
                args_str.append(f'{op_var}_type{count}')
                count += 1
            else:
                args_str.append(str(args[i]))
        return src_code, args_str

    @staticmethod
    def mul(name, node, x, y):
        (x_node, y_node) = node.args
        if isinstance(y_node, torch.fx.node.Node):
            src_code = f"""
                           auto {name} = op::Mul("{name}")
                             .set_input_x1({x})
                             .set_input_x2({y});
                        """
        else:
            # y is scalar
            dtype = node.meta['val'].dtype
            not_support_type = False
            try:
                cpp_dtype = get_cpp_dtype(dtype)
                ascend_dtype = get_ascend_dtype(dtype)
            except:
                cpp_dtype = "float"
                ascend_dtype = "ge::DataType::DT_FLOAT"
                not_support_type = True
            src_code = f"""
                           auto {name}_scalar_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ static_cast<{cpp_dtype}>({y}) }});
                           auto {name}_scalar = op::Const("{name}_scalar")
                             .set_attr_value({name}_scalar_tensor);"""
            if not_support_type:
                ascend_dtype = get_ascend_dtype(dtype)
                src_code += f"""
                           auto {name}_scalar_cast = op::Cast("{name}_scalar_cast")
                             .set_input_x({name}_scalar)
                             .set_attr_dst_type({ascend_dtype});
                           auto {name} = op::Mul("{name}")
                             .set_input_x1({x})
                             .set_input_x2({name}_scalar_cast);
                            """
            else:
                src_code += f"""
                           auto {name} = op::Mul("{name}")
                             .set_input_x1({x})
                             .set_input_x2({name}_scalar);
                            """

        return src_code

    @staticmethod
    def add(name, node, x, y):
        (x_node, y_node) = node.args
        out_dtype = node.meta['val'].dtype
        ascend_dtype = get_ascend_dtype(out_dtype)
        if isinstance(y_node, torch.fx.node.Node):
            x_dtype = x_node.meta['val'].dtype
            y_dtype = y_node.meta['val'].dtype
            x_name = x
            y_name = y
            src_code = ''
            
            if x_dtype != out_dtype:
                ascend_dtype = get_ascend_dtype(out_dtype)
                x_name = f'{name}_x_cast'
                src_code += f"""
                            auto {name}_x_cast = op::Cast("{name}_x_cast")
                              .set_input_x({x})
                              .set_attr_dst_type({ascend_dtype});
                          """
            if y_dtype != out_dtype:
                ascend_dtype = get_ascend_dtype(out_dtype)
                y_name = f'{name}_y_cast'
                src_code += f"""
                            auto {name}_y_cast = op::Cast("{name}_y_cast")
                              .set_input_x({y})
                              .set_attr_dst_type({ascend_dtype});
                          """
            src_code += f"""
                           auto {name} = op::AddV2("{name}")
                             .set_input_x1({x_name})
                             .set_input_x2({y_name});
                        """
        else:
            # y is scalar
            cpp_dtype = get_cpp_dtype(out_dtype)
            src_code = f"""
                           auto {name}_scalar_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ static_cast<{cpp_dtype}>({y}) }});
                           auto {name}_scalar = op::Const("{name}_scalar")
                             .set_attr_value({name}_scalar_tensor);
                           auto {name} = op::AddV2("{name}")
                             .set_input_x1({x})
                             .set_input_x2({name}_scalar);
                        """

        return src_code

    @staticmethod
    def sub(name, x, y):
        src_code = f"""
                       auto {name} = op::Sub("{name}")
                         .set_input_x1({x})
                         .set_input_x2({y});
                    """

        return src_code

    @staticmethod
    def relu(name, x):
        src_code = f"""
                       auto {name} = op::Relu("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def silu(name, x):
        src_code = f"""
                       auto {name} = op::Swish("{name}")
                         .set_input_x({x})
                         .set_attr_scale(1.0);
                    """

        return src_code

    @staticmethod
    def transpose(name, node, input, dim0, dim1):
        input_shape = list(node.args[0].meta['val'].shape)
        output_shape = list(node.meta['val'].shape)
        
        if input_shape == output_shape:
            src_code = f'''
                      auto {name} = op::Identity("{name}")
                        .set_input_x({input});
            '''
            return src_code
        shape_size = len(output_shape)
        shape_str = '{' + ','.join(map(str, output_shape)) + '}' 

        if dynamic_shape_str(shape_str):
            src_code = process_shape_str(shape_str, name)
        else:
            src_code = f"""auto {name}_reshape_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {shape_str});
                           auto {name}_preprocess = op::Const("{name}_preprocess")
                           .set_attr_value({name}_reshape_tensor);
                        """
        src_code += f"""
                       auto {name} = op::Reshape("{name}")
                         .set_input_x({input})
                         .set_input_shape({name}_preprocess);
                    """
        return src_code

    @staticmethod
    def reciprocal(name, x):
        src_code = f"""
                       auto {name} = op::Reciprocal("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def sqrt(name, x):
        src_code = f"""
                       auto {name} = op::Sqrt("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def rsqrt(name, x):
        src_code = f"""
                       auto {name} = op::Rsqrt("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def convolution(name, node, input, weight, bias, stride, padding,
                    dilation, transposed, output_padding, groups):
        assert transposed == 'false'
        assert output_padding == '{0, 0}'
        stride = stride.strip('{}').split(', ')
        padding = padding.strip('{}').split(', ')
        dilation = dilation.strip('{}').split(', ')
        
        if len(stride) == 2:
            real_stride = [1, 1, stride[0], stride[1]]
            stride_str = '{' + ', '.join(map(str, real_stride)) + '}'
        else:
            stride_str = '{' + ', '.join(map(str, stride)) + '}'

        if len(padding) == 2:
            real_padding = [padding[0], padding[0], padding[1], padding[1]]
            padding_str = '{' + ', '.join(map(str, real_padding)) + '}'
        else:
            padding_str = '{' + ', '.join(map(str, padding)) + '}'

        if len(dilation) == 2:
            real_dialtion = [dilation[0], dilation[0], dilation[1], dilation[1]]
            dilation_str = '{' + ', '.join(map(str, real_dialtion)) + '}'
        else:
            dilation_str = '{' + ', '.join(map(str, dilation)) + '}'

        
        format = "NCHW" if node.meta['val'].stride()[-1] == 1 else "NHWC"
        src_code = f"""
                       auto {name} = op::Conv2D("{name}")
                         .set_input_x({input})
                         .set_input_filter({weight})
                         .set_attr_strides({stride_str})
                         .set_attr_pads({padding_str})
                         .set_attr_dilations({dilation_str})
                         .set_attr_groups({groups})
                         .set_attr_data_format("{format}");
                    """

        if bias != 'None':
            src_code += f'{name}.set_input_bias({bias});\n'
        return src_code

    @staticmethod
    def convert_element_type(name, x, torch_dtype, layout=torch.strided, device='cpu'):
        ascend_dtype = get_ascend_dtype(torch_dtype)
        src_code = f"""
                       auto {name} = op::Cast("{name}")
                         .set_input_x({x})
                         .set_attr_dst_type({ascend_dtype});
                    """

        return src_code

    @staticmethod
    def mean(name, x, dims='{}', keepdim='false'):
        src_code = f"""
                       std::vector<int> {name}_axes_value {dims};
                       std::vector<int64_t> {name}_axes_tensor_shape;
                       if ({name}_axes_value.size() != 0) {{
                         {name}_axes_tensor_shape.push_back({name}_axes_value.size());
                       }}
                       auto {name}_axes_tensor = genTensor({name}_axes_tensor_shape, FORMAT_ND, DT_INT32);
                       setTensorData({name}_axes_tensor, reinterpret_cast<uint8_t*>({name}_axes_value.data()), {name}_axes_value.size() * sizeof(int), "{name}_axes");
                       auto {name}_axes = op::Const("{name}_axes")
                         .set_attr_value({name}_axes_tensor);
                       auto {name} = op::ReduceMean("{name}")
                         .set_input_x({x})
                         .set_input_axes({name}_axes)
                         .set_attr_keep_dims({keepdim});
                    """

        return src_code

    @staticmethod
    def symsize(name, x, dim):
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
        # y is scalar
        cpp_dtype = "float"
        ascend_dtype = "ge::DataType::DT_FLOAT"
        src_code = f"""
                        auto {name}_scalar_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ static_cast<{cpp_dtype}>({y}) }});
                        auto {name}_scalar = op::Const("{name}_scalar")
                          .set_attr_value({name}_scalar_tensor);
                        auto {name} = op::Mul("{name}")
                          .set_input_x1({x})
                          .set_input_x2({name}_scalar);
                    """

        return src_code

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
            shape_str = '{' + ', '.join(real_shape) + '}'
        else:
            shape = list(map(str, shape))
            shape_str = '{' + ','.join(shape) + '}'

        if dynamic_shape_str(shape_str):
            src_code = process_shape_str(shape_str, name)
        else:
            src_code = f"""auto {name}_reshape_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {shape_str});
                           auto {name}_preprocess = op::Const("{name}_preprocess") 
                           .set_attr_value({name}_reshape_tensor);
                        """
        src_code += f"""
                       auto {name} = op::Reshape("{name}")
                         .set_input_x({x})
                         .set_input_shape({name}_preprocess);
                    """

        return src_code

    @staticmethod
    def clone(name, x, memory_format=None):
        src_code = f"""
                       auto {name} = op::Identity("{name}")
                         .set_input_x({x});
                    """
        return src_code
      
    @staticmethod
    def _to_copy(name, x, dtype=None, layout=None, device=None):
        if dtype:
            ascend_dtype = get_ascend_dtype(dtype)
            src_code = f"""
                          auto {name} = op::Cast("{name}")
                            .set_input_x({x})
                            .set_attr_dst_type({ascend_dtype});
                        """
        else:
            src_code = f"""
                          auto {name} = op::Identity("{name}")
                            .set_input_x({x});
                        """
        return src_code

    @staticmethod
    def unsqueeze(name, x, dim):
        real_dim_str = dim.replace('[','')
        real_dim_str = real_dim_str.replace(']','')
        real_dim_str = real_dim_str.replace('{','')
        real_dim_str = real_dim_str.replace('}','')
        real_dim_str = '{' + real_dim_str + '}'
        src_code = f"""
                       std::vector<int64_t> {name}_dims{real_dim_str};
                       auto {name} = op::Unsqueeze("{name}")
                         .set_input_x({x})
                         .set_attr_axes({name}_dims);
                    """

        return src_code

    @staticmethod
    def squeeze(name, x, dim):
        if dim == '0':
            real_dim_str = dim
        else:
            real_dim_str = dim.replace('[','')
            real_dim_str = real_dim_str.replace(']','')
            real_dim_str = real_dim_str.replace('{','')
            real_dim_str = real_dim_str.replace('}','')
            real_dim_str = '{' + real_dim_str + '}'
        src_code = f"""
                       auto {name} = op::Squeeze("{name}")
                         .set_input_x({x})
                         .set_attr_axis({real_dim_str});
                    """

        return src_code

    @staticmethod
    def getitem(name, input, index):
        src_code = f"""
                       auto {name} = op::Identity("{name}")
                         .set_input_x({input}, {index});
                    """

        return src_code

    @staticmethod
    def exp(name, x):
        src_code = f"""
                       auto {name} = op::Exp("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def embedding(name, weight, indices):
        src_code = f"""
                       auto {name}_axis_tensor = genTensorWithData<int>({{ 1 }}, FORMAT_NCHW, DT_INT32, {{0}});
                       auto {name}_axis = op::Const("{name}_axis")
                         .set_attr_value({name}_axis_tensor);
                       auto {name} = op::GatherV2("{name}")
                         .set_input_x({weight})
                         .set_input_indices({indices})
                         .set_input_axis({name}_axis);
                    """
        return src_code

    @staticmethod
    def sigmoid(name, x):
        src_code = f"""
                       auto {name} = op::Sigmoid("{name}")
                          .set_input_x({x})
                    """
        return src_code

    @staticmethod
    def pow(name, node, x, exp):
        (x_node, exp_node) = node.args
        if isinstance(exp_node, torch.fx.node.Node):
            src_code = f"""
                          auto {name} = op::Pow("{name}")
                              .set_input_x1({x})
                              .set_input_x2({exp});
                        """
        else:
            # exp is scalar
            dtype = node.meta['val'].dtype
            not_support_type = False
            try:
                cpp_dtype = get_cpp_dtype(dtype)
                ascend_dtype = get_ascend_dtype(dtype)
            except:
                cpp_dtype = "float"
                ascend_dtype = "ge::DataType::DT_FLOAT"
                not_support_type = True
            src_code = f"""
                           auto {name}_scalar_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ static_cast<{cpp_dtype}>({exp}) }});
                           auto {name}_scalar = op::Const("{name}_scalar")
                             .set_attr_value({name}_scalar_tensor);
                           """
            if not_support_type:
                ascend_dtype = get_ascend_dtype(dtype)
                src_code += f"""
                           auto {name}_scalar_cast = op::Cast("{name}_scalar_cast")
                             .set_input_x({name}_scalar)
                             .set_attr_dst_type({ascend_dtype});
                           auto {name} = op::Pow("{name}")
                             .set_input_x1({x})
                             .set_input_x2({name}_scalar_cast);
                           """
            else:
                src_code += f"""
                           auto {name} = op::Pow("{name}")
                             .set_input_x1({x})
                             .set_input_x2({name}_scalar);
                           """
        return src_code

    @staticmethod
    def div(name, node, x, y):
        (x_node, y_node) = node.args
        if isinstance(y_node, torch.fx.node.Node):
            src_code = f"""
                           auto {name} = op::DivNoNan("{name}")
                             .set_input_x1({x})
                             .set_input_x2({y});
                        """
        else:
            div_value = str(1.0 / y_node)
            src_code = f"""
                           auto {name} = op::Muls("{name}")
                             .set_input_x({x})
                             .set_attr_value({div_value});
                        """
        return src_code

    @staticmethod
    def _softmax(name, x, dim='', half_to_float='false'):
        assert(half_to_float == 'false')
        src_code = f"""
                       std::vector<int64_t> {name}_dim{{ {dim} }};
                       auto {name} = op::SoftmaxV2("{name}")
                         .set_input_x({x})
                         .set_attr_axes({name}_dim);
                    """
        return src_code

    @staticmethod
    def sum(name, x, axes='{}', keep_dims='false'):
        src_code = f"""
                       std::vector<int64_t> {name}_axes{axes};
                       auto {name} = op::ReduceSumD("{name}")
                         .set_input_x({x})
                         .set_attr_axes({name}_axes)
                         .set_attr_keep_dims({keep_dims});
                    """

        return src_code

    @staticmethod
    def amax(name, x, axes, keep_dims):
        src_code = f"""
                       auto {name} = op::ReduceMaxD("{name}")
                         .set_input_x({x})
                         .set_attr_axes({axes})
                         .set_attr_keep_dims({keep_dims});
                    """
        return src_code

    @staticmethod
    def permute(name, x, order):
        src_code = f"""
                       auto {name} = op::Permute("{name}")
                         .set_input_x({x})
                         .set_attr_order({order});
                    """
        return src_code

    @staticmethod
    def max_pool2d_with_indices(name, x, ksize, strides, padding='{0, 0}'):
        assert len(ksize.split(',')) == 2
        assert len(strides.split(',')) == 2

        ksize_str = '{1, 1, ' + ksize.strip('{}') + '}' 
        strides_str = '{1, 1,' + strides.strip('{}') + '}'
        paddings = list(map(int, padding.strip('{}').split(',')))

        if paddings != [0, 0]:
            padding0 = str(paddings[0])
            padding1 = str(paddings[1])
            padding_str = f'0, 0, 0, 0, {padding0}, {padding0}, {padding1}, {padding1}'
            src_code = f'''
                           auto {name}_pad_tensor = genTensorWithData<{int}>({{4, 2}}, FORMAT_NCHW, DT_INT32, {{ {padding_str} }});

                           auto {name}_paddings = op::Const("{name}_paddings")
                             .set_attr_value({name}_pad_tensor);
                           auto {name}_pad = op::PadV3("{name}_pad")
                             .set_input_x({x})
                             .set_input_paddings({name}_paddings);
                           auto {name}_fwd_out = op::MaxPool("{name}_fwd_out")
                             .set_input_x({name}_pad)
                             .set_attr_ksize({ksize_str})
                             .set_attr_strides({strides_str})
                             .set_attr_padding("VALID")
                             .set_attr_data_format("NCHW");
                           '''
        else:
            src_code = f'''auto {name}_fwd_out = op::MaxPool("{name}_fwd_out")
                             .set_input_x({x})
                             .set_attr_ksize({ksize_str})
                             .set_attr_strides({strides_str})
                             .set_attr_padding("VALID")
                             .set_attr_data_format("NCHW");
                           '''
        src_code += f'''   auto {name}_shape = op::Shape("{name}_shape")        
                             .set_input_x({name}_fwd_out);

                           auto {name}_indice = op::Empty("{name}_indice")
                             .set_input_shape({name}_shape)
                             .set_attr_dtype(DT_INT64);

                           auto {name} = op::IdentityN("{name}")
                             .create_dynamic_input_x(2)
                             .set_dynamic_input_x(0, {name}_fwd_out)
                             .set_dynamic_input_x(1, {name}_indice)
                             .create_dynamic_output_y(2);
                           '''
        return src_code

    @staticmethod
    def addmm(name, c, a, b, beta='1', alpha='1'):
        src_code = f"""
                       auto {name}_beta_tensor = genTensorWithData<float>({{}}, FORMAT_ND, DT_FLOAT, {{ {beta} }});
                       auto {name}_alpha_tensor = genTensorWithData<float>({{}}, FORMAT_ND, DT_FLOAT, {{ {alpha} }});

                       auto {name}_beta = op::Const("{name}_beta")
                         .set_attr_value({name}_beta_tensor);
                       auto {name}_alpha = op::Const("{name}_alpha")
                         .set_attr_value({name}_alpha_tensor);

                       auto {name}_c_beta = op::Mul("{name}_c_beta")
                         .set_input_x1({c})
                         .set_input_x2({name}_beta);

                       auto {name}_a_alpha = op::Mul("{name}_a_alpha")
                         .set_input_x1({a})
                         .set_input_x2({name}_alpha);

                       auto {name}_matmul = op::MatMul("{name}_matmul")
                         .set_input_x1({name}_a_alpha)
                         .set_input_x2({b});

                       auto {name} = op::AddV2("{name}")
                         .set_input_x1({name}_c_beta)
                         .set_input_x2({name}_matmul);
                    """

        return src_code

    @staticmethod
    def var(name, x, axes, correction='1', keepdim='true'):
        if correction == '1':
            unbiased = 'true'
        elif correction == 0:
            unbiased = 'false'
        else:
            raise RuntimeError("not supported yet!")
        
        src_code = f"""
                       // 1. mean
                       std::vector<int> {name}_axes_value {axes};
                       std::vector<int64_t> {name}_axes_tensor_shape;
                       if ({name}_axes_value.size() != 0) {{
                         {name}_axes_tensor_shape.push_back({name}_axes_value.size());
                       }}
                       auto {name}_axes_tensor = genTensor({name}_axes_tensor_shape, FORMAT_ND, DT_INT32);
                       setTensorData({name}_axes_tensor, reinterpret_cast<uint8_t*>({name}_axes_value.data()), {name}_axes_value.size() * sizeof(int), "{name}_axes");
                       auto {name}_axes = op::Const("{name}_axes")
                         .set_attr_value({name}_axes_tensor);
                       auto {name}_mean = op::ReduceMean("{name}_mean")
                         .set_input_x({x})
                         .set_input_axes({name}_axes)
                         .set_attr_keep_dims({keepdim});
    
                       // 2. broadcast to self
                       auto {name}_input_shape = op::Shape("{name}_input_shape")
                         .set_input_x({x});
                       auto {name}_broadcast_to = op::BroadcastTo("{name}_broadcast_to")
                         .set_input_x({name}_mean)
                         .set_input_shape({name}_input_shape);
        
                       // 3. ReduceStdV2Update
                       auto {name} = op::ReduceStdV2Update("{name}")
                         .set_input_x({x})
                         .set_input_mean({name}_broadcast_to)
                         .set_attr_dim({axes})
                         .set_attr_unbiased({unbiased})
                         .set_attr_keepdim({keepdim});
                    """

        return src_code

    @staticmethod
    def log(name, x):
        src_code = f"""
                       auto {name} = op::Log("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def gather(name, x, dim, index):
        src_code = f"""
                       auto {name}_dim_tensor = genTensorWithData<int>({{1}}, FORMAT_NCHW, DT_INT32, {{ {dim} }});
                       auto {name}_dim = op::Const("{name}_dim")
                         .set_attr_value({name}_dim_tensor);
                       auto {name} = op::GatherD("{name}")
                         .set_input_x({x})
                         .set_input_dim({name}_dim)
                         .set_input_index({index})
                         .set_attr_dim({dim});
                    """

        return src_code

    @staticmethod
    def neg(name, x):
        src_code = f"""
                       auto {name} = op::Neg("{name}")
                         .set_input_x({x});
                    """

        return src_code

    @staticmethod
    def expand(name, node, x, shape):
        x_shape = list(node.target.x.node.meta['val'].shape)
        y_shape = list(node.meta['val'].shape)
        if x_shape == y_shape:
            src_code = f"""
                           auto {name} = op::Identity("{name}")
                             .set_input_x({x});
                        """
            return src_code

        src_code = f"""
                       std::vector<int64_t> {name}_shape{shape};
                       auto {name} = op::ExpandD("{name}")
                         .set_input_x({x})
                         .set_attr_shape({name}_shape);
                    """
        return src_code

    @staticmethod
    def zeros_like(name, x, value):
        # TODO(tangzhiyi): ignore kwargs, need to check this
        src_code = f"""
                       auto {name} = op::ZerosLike("{name}")
                         .set_input_x({x});
                    """

        return src_code


    @staticmethod
    def full(name, node, dims, fill_value):
        dims = dims.strip('{}').split(', ')
        if len(dims) == 0:
            dims = [1]
        torch_dtype = node.kwargs['dtype']
        dims_str = '{' + ','.join(map(str, dims)) + '}'
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        src_code = f"""
                    std::vector<int> {name}_axes_value {dims_str};
                    std::vector<int64_t> {name}_axes_tensor_shape;
                    if ({name}_axes_value.size() != 0) {{
                      {name}_axes_tensor_shape.push_back({name}_axes_value.size());
                    }}
                    auto {name}_axes_tensor = genTensor({name}_axes_tensor_shape, FORMAT_ND, DT_INT32);
                    setTensorData({name}_axes_tensor, reinterpret_cast<uint8_t*>({name}_axes_value.data()), {name}_axes_value.size() * sizeof(int), "{name}_axes");
                    auto {name}_axes = op::Const("{name}_axes")
                      .set_attr_value({name}_axes_tensor);

                    auto {name}_val_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ {fill_value} }});
                    auto {name}_val = op::Const("{name}_val")
                      .set_attr_value({name}_val_tensor);

                    auto {name} = op::Fill("{name}")
                      .set_input_dims({name}_axes)
                      .set_input_value({name}_val);
                    """

        return src_code


    @staticmethod
    def scatter(name, node, var, dim, index, value):
        assert(len(node.args) > 3)
        value_node = node.args[3]
        if isinstance(value_node, torch.fx.node.Node):
            src_code = f"""
                           auto {name} = op::ScatterElements("{name}")
                             .set_input_data({var})
                             .set_input_indices({index})
                             .set_input_updates({value})
                             .set_attr_axis({dim});
                        """
        else:
            dtype = node.meta['val'].dtype
            ascend_dtype = get_ascend_dtype(dtype)
            cpp_dtype = get_cpp_dtype(dtype)
            src_code = f"""
                           auto {name}_value_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ {value} }});
                           auto {name}_value = op::Const("{name}_value")
                             .set_attr_value({name}_value_tensor);
                           auto {name}_index_shape = op::Shape("{name}_index_shape")
                             .set_input_x({index});
                           auto {name}_value_bcast = op::BroadcastTo("{name}_value_bcast")
                             .set_input_x({name}_value)
                             .set_input_shape({name}_index_shape);
                           auto {name} = op::ScatterElements("{name}")
                             .set_input_data({var})
                             .set_input_indices({index})
                             .set_input_updates({name}_value_bcast)
                             .set_attr_axis({dim});
                        """

        return src_code

    @staticmethod
    def mm(name, x, y):
        src_code = f"""
                       auto {name} = op::MatMul("{name}")
                         .set_input_x1({x})
                         .set_input_x2({y});
                    """

        return src_code

    @staticmethod
    def bmm(name, x, y):
        src_code = f"""
                       auto {name} = op::BatchMatMul("{name}")
                         .set_input_x1({x})
                         .set_input_x2({y});
                    """

        return src_code

    @staticmethod
    def convolution_backward(name, grad_output, input, weight, bias_size,
                             stride, padding, dilation, transposed, output_padding,
                             groups, grad_input_mask):
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

        src_code = f"""
                       // 1. broadcast
                       auto {name}_shape = op::Shape("{name}_cond_shape")
                         .set_input_x({cond});
                       auto {name}_x1_bcast = op::BroadcastTo("{name}_x1_bcast")
                         .set_input_x({x1})
                         .set_input_shape({name}_shape);
                       auto {name}_x2_bcast = op::BroadcastTo("{name}_x2_bcast")
                         .set_input_x({x2})
                         .set_input_shape({name}_shape);
                       auto {name} = op::Select("{name}")
                         .set_input_condition({cond})
                         .set_input_x1({name}_x1_bcast)
                         .set_input_x2({name}_x2_bcast);
                    """

        return src_code

    @staticmethod
    def le(name, node, x1, x2):
        (x1_node, x2_node) = node.args
        if isinstance(x2_node, torch.fx.node.Node):
            src_code = f"""
                           auto {name} = op::LessEqual("{name}")
                             .set_input_x1({x1})
                             .set_input_x2({x2});
                        """
        else:
            # TODO(tangzhiyi): get value type, now assume float
            src_code = f"""
                           auto {name}_x2_tensor = genTensorWithData<float>({{}}, FORMAT_NCHW, DT_FLOAT, {{ {x2} }});
                           auto {name}_x2 = op::Const("{name}_x2")
                             .set_attr_value({name}_x2_tensor);
                           auto {name} = op::LessEqual("{name}")
                             .set_input_x1({x1})
                             .set_input_x2({name}_x2);
                        """

        return src_code

    @staticmethod
    def scalar_tensor(name, node, val):
        torch_dtype = node.kwargs['dtype']
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        src_code = f"""
                       auto {name}_val_tensor = genTensorWithData<{cpp_dtype}>({{}}, FORMAT_NCHW, {ascend_dtype}, {{ {val} }});
                       auto {name} = op::Const("{name}")
                         .set_attr_value({name}_val_tensor);
                    """

        return src_code

    @staticmethod
    def ret_tuple(name, in1, in2):
        src_code = f"""
                       auto {name} = op::IdentityN("{name}")
                         .create_dynamic_input_x(2)
                         .set_dynamic_input_x(0, {in1})
                         .set_dynamic_input_x(1, {in2})
                         .create_dynamic_output_y(2);
                    """

        return src_code

    @staticmethod
    def ret_triple(name, in1, in2, in3):
        src_code = f"""
                       auto {name} = op::IdentityN("{name}")
                         .create_dynamic_input_x(3)
                         .set_dynamic_input_x(0, {in1})
                         .set_dynamic_input_x(1, {in2})
                         .set_dynamic_input_x(2, {in3})
                         .create_dynamic_output_y(3);
                    """

        return src_code

    @staticmethod
    def t(name, node, input):
        shape = node.meta['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape.reverse()
        order_str = '{' + ','.join(map(str, permute_shape)) + '}'
        src_code = f"""auto {name} = op::Permute("{name}")
                         .set_input_x({input})
                         .set_attr_order({order_str});
                    """
        return src_code

    @staticmethod
    def log_softmax(name, x, dim, half_to_float):
        assert half_to_float == 'false'
        src_code = f"""auto {name} = op::LogSoftmaxV2("{name}")
                         .set_input_logits({x})
                         .set_attr_axes({{ {dim}  }});
                    """
        return src_code

    @staticmethod
    def log_softmax_backward(name, grad_output, output, dim, input_dtype):
        src_code = f"""auto {name} = op::LogSoftmaxGrad("{name}")
                         .set_input_grad({grad_output})
                         .set_input_x({output})
                         .set_attr_axis({{ {dim}  }});
                    """
        return src_code

    @staticmethod
    def nll_loss_forward(name, node, x, target, weight, reduction, ignore_index):
        assert weight == 'None'
        assert ignore_index == '-100'
        reduction_str = get_reduction_str(reduction)
        csize = str(list(node.target.x.node.meta['val'].shape)[1])
        src_code = f"""std::vector<int64_t> {name}_weight_dims {{ {csize} }};
                       auto {name}_target_cast = op::Cast("{name}_target_cast")
                         .set_input_x({target})
                         .set_attr_dst_type(DT_INT32);
                       auto {name}_weight = op::FillV2D("{name}_weight")
                         .set_attr_value(1.0)
                         .set_attr_dims({name}_weight_dims);
                       auto {name} = op::NLLLoss("{name}")
                         .set_input_x({x})
                         .set_input_target({name}_target_cast)
                         .set_input_weight({name}_weight)
                         .set_attr_reduction("{reduction_str}")
			                   .set_attr_ignore_index({ignore_index});
		                """
        return src_code

    @staticmethod
    def native_batch_norm_legit_functional(name, x, weight, bias, running_mean,
                                           running_var, train, momentum, eps, node):
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
        csize = str(list(node.target.x.node.meta['val'].shape)[1])

        src_code = f"""std::vector<int64_t> {name}_weight_dims {{ {csize} }};
                       auto {name}_target_cast = op::Cast("{name}_target_cast")
                         .set_input_x({target})
                         .set_attr_dst_type(DT_INT32);
                       auto {name}_weight = op::FillV2D("{name}_weight")
                         .set_attr_value(1.0)
                         .set_attr_dims({name}_weight_dims);
                       auto {name} = op::NLLLossGrad("{name}")
                         .set_input_x({x})
                         .set_input_y_grad({grad_output})
                         .set_input_target({name}_target_cast)
                         .set_input_weight({name}_weight)
                         .set_input_total_weight({total_weight})
                         .set_attr_reduction("{reduction_str}")
                         .set_attr_ignore_index({ignore_index});
                       """
        return src_code

    @staticmethod
    def threshold_backward(name, grad_output, x, threshold):
        if threshold == '0':
            src_code = f"""auto {name} = op::ReluGrad("{name}")
                             .set_input_gradients({grad_output})
                             .set_input_features({x});
                           """
        else:
            src_code = f"""auto {name} = op::ThresholdGradV2D("{name}")
                             .set_input_gradients({grad_output})
                             .set_input_features({x})
                             .set_attr_threshold({threshold});
                           """
        return src_code

    @staticmethod
    def zeros_like(name, x, *args):
        src_code = f"""auto {name} = op::ZerosLike("{name}")
                         .set_input_x({x});
                    """
        return src_code

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

        src_code = f"""auto {name}_split = op::SplitD("{name}_split")
                         .set_input_x({x})
                         .set_attr_split_dim({dim})
                         .set_attr_num_split(2)
                         .create_dynamic_output_y(2);
                       auto {name}_complex = op::Complex("{name}_complex")
                           .set_input_real({name}_split, 0)
                           .set_input_imag({name}_split, 1)
                           .set_attr_Tout({output_dtype});
                  """
        if dynamic_shape_str(shape_str):
            src_code += process_shape_str(shape_str, name)
        else:
            src_code += f"""auto {name}_reshape_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {shape_str});
                            auto {name}_preprocess = op::Const("{name}_preprocess") 
                            .set_attr_value({name}_reshape_tensor);
                         """
        src_code += f"""
                       auto {name} = op::Reshape("{name}")
                         .set_input_x({name}_complex)
                         .set_input_shape({name}_preprocess);
                  """
        return src_code
      
    @staticmethod
    def view_as_real(name, node, x):
        assert node.meta['val'].dtype == torch.float32
        x_shape = list(node.target.x.node.meta['val'].shape)
        dim = len(x_shape) - 1

        src_code = f"""auto {name}_real = op::Real("{name}_real")
                           .set_input_input({x});
                       auto {name}_imag = op::Imag("{name}_imag")
                           .set_input_input({x});
                       auto {name} = op::Pack("{name}_pack")
                           .create_dynamic_input_x(2)
                           .set_dynamic_input_x(0, {name}_real)
                           .set_dynamic_input_x(1, {name}_imag)
                           .set_attr_axis({dim})
                           .set_attr_N(2);
                    """
        return src_code

    @staticmethod
    def slice(name, node, x, dim, start, end):
        # TODO(tangzhiyi): miss step parameter
        x_shape = list(node.target.x.node.meta['val'].shape)
        y_shape = list(node.meta['val'].shape)

        dim = int(dim)
        start = int(start)
        assert dim >= 0 and dim < len(x_shape)
        assert start >=0 and start < x_shape[dim]
        
        offset = ['0'] * len(x_shape)
        offset[dim] = str(start)

        shape_size = '{' + str(len(x_shape)) + '}'
        offset_str = '{' + ','.join(offset) + '}'
        size_str = '{' + ','.join(map(str, y_shape)) + '}'
        
        # NB: In some cases, this function may obtain an incorrect value.
        # example:
        # >>def fn(x):
        # >>    x = x[:, 1:3]
        # >>    return x
        # >>opt_model = torch.compile(fn, backend='ascendgraph')
        # >>x = torch.tensor([[1, 2, 3],
        #                    [4, 5, 6],
        #                    [7, 8, 9]], dtype=torch.float32)
        #
        # output is : tensor([[1., 2.],
        #                     [3., 4.],
        #                     [5., 6.]])
        # right output is : tensor([[2., 3.],
        #                     [5., 6.],
        #                     [8., 9.]])
        # Kernel output is right, but aot_autograd changed it when process alis in
        # torch/_functorch/aot_autograd.py(530)gen_alias_from_base().
        # TODO(tangzhiyi): fix this
        if dynamic_shape_str(offset_str):
            src_code = process_shape_str(offset_str, name, '_offset')
        else:
            src_code = f"""auto {name}_offset_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {offset_str});
                           auto {name}_preprocess_offset = op::Const("{name}_preprocess_offset")
                           .set_attr_value({name}_offset_tensor);
                        """
        if dynamic_shape_str(size_str):
            src_code += process_shape_str(size_str, name, '_size')
        else:
            src_code += f"""auto {name}_size_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {size_str});
                            auto {name}_preprocess_size = op::Const("{name}_preprocess_size")
                            .set_attr_value({name}_size_tensor);
                        """
        src_code += f"""
                       auto {name} = op::Slice("{name}")
                           .set_input_x({x})
                           .set_input_offsets({name}_preprocess_offset)
                           .set_input_size({name}_preprocess_size);
                    """
        return src_code

    @staticmethod
    def cat(name, node, x, dim=0):
        x_list = x.strip('{}').split(', ')
        x_size = len(x_list)
        y_dtype = node.meta['val'].dtype
        
        x_node = node.args[0]
        x_names = []
        src_code = ''
        for i, v in enumerate(x_node):
            dtype = v.meta['val'].dtype
            if dtype == y_dtype:
                x_names.append(x_list[i])
                continue
            
            # cast to y_dtype
            ascend_dtype = get_ascend_dtype(y_dtype)
            src_code += f'''
                           auto {name}_cast_{i} = op::Cast("{name}_cast_{i}")
                             .set_input_x({x_list[i]})
                             .set_attr_dst_type({ascend_dtype});
            '''
            x_names.append(f'{name}_cast_{i}')
            import pdb;pdb.set_trace()
        src_code = f"""auto {name} = op::ConcatD("{name}")
                         .create_dynamic_input_x({x_size})"""
        
        for i, x in enumerate(x_names):
            src_code += f""".set_dynamic_input_x({i}, {x})"""
        src_code += f""".set_attr_concat_dim({dim})
                        .set_attr_N({x_size});
                    """
        return src_code
      
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

        shape_size = '{' + str(len(x_shape)) + '}'
        offset_str = '{' + ','.join(map(str, offset)) + '}'
        size_str = '{' + ','.join(map(str, size)) + '}'
        y_shape_size = '{' + str(len(y_shape)) + '}'
        y_shape_str = '{' + ','.join(map(str, y_shape)) + '}'

        if dynamic_shape_str(offset_str):
            src_code = process_shape_str(offset_str, name, '_offset')
        else:
            src_code = f"""auto {name}_offset_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {offset_str});
                           auto {name}_preprocess_offset = op::Const("{name}_preprocess_offset")
                           .set_attr_value({name}_offset_tensor);
                        """
        if dynamic_shape_str(size_str):
            src_code += process_shape_str(size_str, name, '_size')
        else:
            src_code += f"""auto {name}_size_tensor = genTensorWithData<int>({{ {shape_size} }}, FORMAT_ND, DT_INT32, {size_str});
                            auto {name}_preprocess_size = op::Const("{name}_preprocess_size")
                            .set_attr_value({name}_size_tensor);
                        """
        src_code += f"""
                       auto {name}_slice = op::Slice("{name}_slice")
                           .set_input_x({x})
                           .set_input_offsets({name}_preprocess_offset)
                           .set_input_size({name}_preprocess_size);
                       
                       auto {name}_reshape_tensor = genTensorWithData<int>({y_shape_size}, FORMAT_ND, DT_INT32, {y_shape_str});
                       auto {name}_reshape = op::Const("{name}_reshape") 
                         .set_attr_value({name}_reshape_tensor);

                       auto {name} = op::Reshape("{name}")
                         .set_input_x({name}_slice)
                         .set_input_shape({name}_reshape);
                       """
        return src_code