import torch

from typing import Any
from torch.fx.node import Node

from .cpp_code_template import fixed_kernel_code

graph_id = 0

def get_graph_id():
    global graph_id
    graph_id = graph_id + 1
    return graph_id


class AscendCodegen(torch.fx.Interpreter):
    def __init__(self, graph):

        self.graph = graph
        self.override = AscendOverrides
        self.code = fixed_kernel_code

        self.graph_id = str(get_graph_id())
        self.input_args = []
        self.output_args = []
        self.build_graph = ''

        super().__init__(graph)

    def placeholder(self, n: Node):
        self.input_args.append(n)
        fake_tensor = n.meta['val']
        op_str = getattr(self.override, "data")(n.name, fake_tensor.dtype, fake_tensor.shape)
        self.build_graph += op_str

    def call_function(self, n: Node):
        target = n.target
        name = n.name

        if hasattr(target, "name"):
            real_op = target.name().split('::')[-1]
            if real_op.find('.') != -1:
                real_op = real_op.split('.')[0]
        else:
            real_op = name.rsplit('_', 1)[0] if name[-1].isdigit() else name

        op_str = getattr(self.override, real_op)(n)
        self.build_graph += op_str

    def call_method(self, n: Node):
        pass

    def output(self, n: Node):
        for arg in n.args:
            self.output_args.extend(arg)

    def run_node(self, n : Node) -> Any:
        assert isinstance(n.args, tuple)
        assert isinstance(n.kwargs, dict)
        return getattr(self, n.op)(n) 

    def codegen(self):
        self.run()

        input_names = [x.name for x in self.input_args]
        output_names = []
        for x in self.output_args:
            if isinstance(x, torch.fx.node.Node):
                output_names.append(x.name)
            else:
                output_names.append(str(x))
        output_args_nodes = [x for x in self.output_args if x not in self.input_args and x is not None]

        py_outputs = []
        unique_output_names = []
        for i, node in enumerate(output_args_nodes):
            if node.name not in unique_output_names:
                unique_output_names.append(node.name)
            else:
                continue

            name = node.name
            val = node.meta['val']
            shape = str(tuple(val.size()))
            stride = str(tuple(val.stride()))
            device = val.device.type
            dtype = str(val.dtype)

            index = len(unique_output_names) - 1
            code_str = f'''{name} = torch.from_numpy(output_np[{index}])'''
            py_outputs.append(code_str)

            code_str = f'''    graph_outputs.push_back({name});
'''
            self.build_graph += code_str

        py_rand_inputs = []
        for node in self.input_args:
            name = node.name
            val = node.meta['val']
            shape = str(tuple(val.size()))
            stride = str(tuple(val.stride()))
            device = val.device.type
            dtype = str(val.dtype)
            code_str = f'''{name} = rand_strided({shape}, {stride}, device='{device}', dtype={dtype})'''
            py_rand_inputs.append(code_str)      

        del_args = ['del ' + x for x in input_names if x not in output_names]
        kernel_code = fixed_kernel_code
        self.code = kernel_code.format(graph_id=self.graph_id,
                                       build_graph=self.build_graph,
                                       py_inputs=', '.join(input_names),
                                       py_outputs='\n    '.join(py_outputs),
                                       py_returns=', '.join(output_names),
                                       delete_unuse_inputs='\n    '.join(del_args),
                                       py_rand_inputs='\n    '.join(py_rand_inputs))
        return self.code


def get_ascend_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "ge::DataType::DT_INT64"
    elif dtype == torch.float32:
        return "ge::DataType::DT_FLOAT"
    elif dtype == torch.int32:
        return "ge::DataType::DT_INT32"
    else:
        raise RuntimeError("unknow torch data tyep type in get_ascend_dtype!")


def get_cpp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "int64_t"
    elif dtype == torch.float32:
        return "float"
    else:
        raise RuntimeError("unknow torch data tyep type in get_cpp_dtype!")


class AscendOverrides:
    """Map element-wise ops to Ascend C++"""

    @staticmethod
    def output(args):
        output_str = ''
        for arg in args:
            arg_name = arg.name if arg is isinstance(arg, torch.fx.node.Node) else arg
            output_str += f'''
    graph_outputs.push_back({arg_name});'''
        return output_str

    @staticmethod
    def data(name, dtype, shape):
        tensor_format = "FORMAT_NCHW"
        tensor_shape = '{' + ','.join(list(map(str, shape))) + '}'
        ascend_dtype = get_ascend_dtype(dtype)
        data_op = f'''
    std::vector<int64_t> {name}_tensor_shape{tensor_shape};
    TensorDesc {name}_tensor_desc_data_op = TensorDesc(ge::Shape({name}_tensor_shape), {tensor_format}, {ascend_dtype});
    auto {name} = op::Data("{name}");
    {name}.update_input_desc_x({name}_tensor_desc_data_op);
    {name}.update_output_desc_y({name}_tensor_desc_data_op);
    graph.AddOp({name});
    graph_inputs.push_back({name});
'''
        return data_op

    @staticmethod
    def mul(n: Node):
        name = n.name
        (x, y) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        if isinstance(y, torch.fx.node.Node):
            y_name = y.name if isinstance(y, torch.fx.node.Node) else y
            mul_op = f'''
        auto {name} = op::Mul("{name}")
            .set_input_x1({x_name})
            .set_input_x2({y_name});
        graph.AddOp({name});
'''
        else:
            # y is scalar
            cpp_dtype = get_cpp_dtype(n.meta['val'].dtype)
            ascend_dtype = get_ascend_dtype(n.meta['val'].dtype)
            mul_op = f'''
    {cpp_dtype} {name}_scalar_value = static_cast<{cpp_dtype}>({y});
    auto {name}_scalar_tensor = genTensor(std::vector<int64_t>(), FORMAT_NCHW, {ascend_dtype});
    setTensorData({name}_scalar_tensor, reinterpret_cast<uint8_t*>(&{name}_scalar_value), sizeof({cpp_dtype}), "{name} scalar");
    auto {name}_scalar = op::Const("{name}_scalar")
        .set_attr_value({name}_scalar_tensor);
    auto {name} = op::Mul("{name}")
        .set_input_x1({x_name})
        .set_input_x2({name}_scalar);
    graph.AddOp({name}_scalar);
    graph.AddOp({name});
'''
        return mul_op

    @staticmethod
    def add(n: Node):
        name = n.name
        (x, y) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x

        if isinstance(y, torch.fx.node.Node):
            y_name = y.name
            add_op = f'''
    auto {name} = op::AddV2("{name}")
        .set_input_x1({x_name})
        .set_input_x2({y_name});
    graph.AddOp({name});
'''
        else:
#             # y is scalar
#             y_value = str(y)
#             add_op = f'''
#     auto {name} = op::Adds("{name}")
#         .set_input_x({x_name})
#         .set_attr_value({y_value});
#     graph.AddOp({name});
# '''

            # y is scalar
            cpp_dtype = get_cpp_dtype(n.meta['val'].dtype)
            ascend_dtype = get_ascend_dtype(n.meta['val'].dtype)
            add_op = f'''
    {cpp_dtype} {name}_scalar_value = static_cast<{cpp_dtype}>({y});
    auto {name}_scalar_tensor = genTensor(std::vector<int64_t>(), FORMAT_NCHW, {ascend_dtype});
    setTensorData({name}_scalar_tensor, reinterpret_cast<uint8_t*>(&{name}_scalar_value), sizeof({cpp_dtype}), "{name} scalar");
    auto {name}_scalar = op::Const("{name}_scalar")
        .set_attr_value({name}_scalar_tensor);
    auto {name} = op::AddV2("{name}")
        .set_input_x1({x_name})
        .set_input_x2({name}_scalar);
    graph.AddOp({name}_scalar);
    graph.AddOp({name});
'''

        return add_op

    @staticmethod
    def sub(n: Node):
        name = n.name
        (x, y) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        y_name = y.name if isinstance(y, torch.fx.node.Node) else y
        sub_op = f'''
    auto {name} = op::Sub("{name}")
        .set_input_x1({x_name})
        .set_input_x2({y_name});
    graph.AddOp({name});
'''
        return sub_op

    @staticmethod
    def relu(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        relu_op = f'''
    auto {name} = op::Relu("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return relu_op

    @staticmethod
    def reciprocal(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        reciprocal_op = f'''
    auto {name} = op::Reciprocal("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return reciprocal_op

    @staticmethod
    def sqrt(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        sqrt_op = f'''
    auto {name} = op::Sqrt("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return sqrt_op

    @staticmethod
    def rsqrt(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        rsqrt_op = f'''
    auto {name} = op::Rsqrt("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return rsqrt_op

    @staticmethod
    def convolution(n: Node):
        name = n.name
        (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) = n.args
        input_name = input.name if isinstance(input, torch.fx.node.Node) else input
        weight_name = weight.name if isinstance(weight, torch.fx.node.Node) else weight
        bias_name = bias.name if isinstance(bias, torch.fx.node.Node) else bias

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

        groups_str = str(groups)
        format_str = "NCHW" if n.meta['val'].stride()[-1] == 1 else "NHWC"

        assert transposed is False
        assert output_padding == [0, 0]

        conv2d_op = f'''
    auto {name} = op::Conv2D("{name}")
        .set_input_x({input_name})
        .set_input_filter({weight_name})
        .set_attr_strides({stride_str})
        .set_attr_pads({padding_str})
        .set_attr_dilations({dilation_str})
        .set_attr_groups({groups_str})
        .set_attr_data_format("{format_str}");
'''
        if bias_name is not None:
            conv2d_op += f'''    {name}.set_input_bias({bias_name});
'''
        conv2d_op += f'''    graph.AddOp({name});'''
        return conv2d_op

    @staticmethod
    def convert_element_type(n: Node):
        name = n.name
        (x, torch_dtype) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        ascend_dtype = get_ascend_dtype(torch_dtype)
        cast_op = f'''
    auto {name} = op::Cast("{name}")
        .set_input_x({x_name})
        .set_attr_dst_type({ascend_dtype});
    graph.AddOp({name});
'''
        return cast_op

    @staticmethod
    def mean(n: Node):
        name = n.name

        dims = []
        keepdim = False
        x = None
        if len(n.args) == 3:
            (x, dims, keepdim) = n.args
        elif len(n.args) == 1:
            x = n.args[0]
        else:
            raise RuntimeError("unsupported yet!")
        x_name = x.name
        dims_str = '{' + ', '.join(list(map(str, dims))) + '}'
        keepdim_str = "true" if keepdim else "false"

        mean_op = f'''    
    std::vector<int> {name}_axes_value {dims_str};
    std::vector<int64_t> {name}_axes_tensor_shape;
    if ({name}_axes_value.size() != 0) {{
        {name}_axes_tensor_shape.push_back({name}_axes_value.size());
    }}
    auto {name}_axes_tensor = genTensor({name}_axes_tensor_shape, FORMAT_ND, DT_INT32);
    setTensorData({name}_axes_tensor, reinterpret_cast<uint8_t*>({name}_axes_value.data()), {name}_axes_value.size() * sizeof(int), "{name}_axes");
    auto {name}_axes = op::Const("{name}_axes")
        .set_attr_value({name}_axes_tensor);
    auto {name} = op::ReduceMean("{name}")
        .set_input_x({x_name})
        .set_input_axes({name}_axes)
        .set_attr_keep_dims({keepdim_str});
    graph.AddOp({name}_axes);
    graph.AddOp({name});
'''
        return mean_op

    @staticmethod
    def view(n: Node):
        name = n.name
        (x, size) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x 

        numel = n.meta['val'].numel()
        shape = list(n.meta['val'].shape)
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

        view_op = f'''
    auto {name} = op::TransShape("{name}")
        .set_input_x({x_name})
        .set_attr_outShape({shape_str});
    graph.AddOp({name});
'''
        return view_op

    @staticmethod
    def clone(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        clone_op = f'''
    auto {name} = op::Identity("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return clone_op

    @staticmethod
    def unsqueeze(n: Node):
        name = n.name
        (x, dim) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        real_dim_str = str(dim)
        real_dim_str = real_dim_str.replace('[','')
        real_dim_str = real_dim_str.replace(']','')
        real_dim_str = '{' + real_dim_str + '}'
        unsqueeze_op = f'''
    std::vector<int64_t> {name}_dims{real_dim_str};
    auto {name} = op::Unsqueeze("{name}")
        .set_input_x({x_name})
        .set_attr_axes({name}_dims);
    graph.AddOp({name});
'''
        return unsqueeze_op

    @staticmethod
    def squeeze(n: Node):
        name = n.name
        (x, dim) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        if dim == 0:
            dim_str = '0'
        else:
            dim_str = str(dim)
            dim_str = dim_str.replace('[','')
            dim_str = dim_str.replace(']','')
            dim_str = '{' + dim_str + '}'
        squeeze_op = f'''
    auto {name} = op::Squeeze("{name}")
        .set_input_x({x_name})
        .set_attr_axis({dim_str});
    graph.AddOp({name});
'''
        return squeeze_op

    @staticmethod
    def getitem(n: Node):
        name = n.name
        (input, index) = n.args
        input_name = input.name

        getitem_op = f'''
    auto {name} = op::Identity("{name}")
        .set_input_x({input_name}, {index});
    graph.AddOp({name});
'''
        return getitem_op

    @staticmethod
    def exp(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        exp_op = f'''
    auto {name} = op::Exp("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return exp_op

    @staticmethod
    def div(n: Node):
        name = n.name
        (x, y) = n.args 
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x

        if isinstance(y, torch.fx.node.Node):
            y_name = y.name
            div_op = f'''
    auto {name} = op::DivNoNan("{name}")
        .set_input_x1({x_name})
        .set_input_x2({y_name});
    graph.AddOp({name});
'''
        else:
            div_value = str(1.0 / y)
            div_op = f'''
    auto {name} = op::Muls("{name}")
        .set_input_x({x_name})
        .set_attr_value({div_value});
    graph.AddOp({name});
'''
        return div_op

    @staticmethod
    def sum(n: Node):
        name = n.name
        keep_dims = False
        axes = []
        if len(n.args) == 1:
            x = n.args[0]
        elif len(n.args) == 3:
            (x, axes, keep_dims) = n.args
        elif len(n.args) == 2:
            (x, axes) = n.args

        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        axes_str = "{" + ','.join(map(str, axes)) + '}'
        keep_dims_str = "true" if keep_dims else "false"

        sum_op = f'''
    std::vector<int64_t> {name}_axes{axes_str};
    auto {name} = op::ReduceSumD("{name}")
        .set_input_x({x_name})
        .set_attr_axes({name}_axes)
        .set_attr_keep_dims({keep_dims_str});
    graph.AddOp({name});
'''
        return sum_op

    @staticmethod
    def amax(n: Node):
        name = n.name
        (x, axes, keep_dims) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        axes_str = "{" + ','.join(map(str, axes)) + '}'
        keep_dims_str = "true" if keep_dims else "false"

        amax_op = f'''
    auto {name} = op::ReduceMaxD("{name}")
        .set_input_x({x_name})
        .set_attr_axes({axes_str})
        .set_attr_keep_dims({keep_dims_str});
    graph.AddOp({name});
'''
        return amax_op

    @staticmethod
    def permute(n: Node):
        name = n.name
        (x, order) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        order_str = '{' + ','.join(map(str, order)) + '}'
        permute_op = f'''
    auto {name} = op::Permute("{name}")
        .set_input_x({x_name})
        .set_attr_order({order_str});
    graph.AddOp({name});
'''
        return permute_op

    @staticmethod
    def max_pool2d_with_indices(n: Node):
        name = n.name
        if len(n.args) == 3:
            (x, ksize, strides) = n.args
            padding = [0, 0]
        elif len(n.args) == 4:
            (x, ksize, strides, padding) = n.args
        else:
            raise RuntimeError("not supproted yet!")
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x

        assert len(ksize) == 2
        assert len(strides) == 2

        ksize_str = '{1 , ' + str(ksize).strip('[]') + ' , 1}'
        strides_str = '{1, ' + str(strides).strip('[]') + ' , 1}'

        if padding != [0, 0]:
            padding0 = str(padding[0])
            padding1 = str(padding[1])
            padding_str = f'0, 0, 0, 0, {padding0}, {padding0}, {padding1}, {padding1}'
            max_pool_op = f'''
    TensorDesc {name}_pad_desc(ge::Shape({{4, 2}}), FORMAT_NCHW, DT_INT32);
    std::vector<int> {name}_pad_value {{ {padding_str} }};
    Tensor {name}_pad_tensor({name}_pad_desc);
    setTensorData({name}_pad_tensor, reinterpret_cast<uint8_t*>({name}_pad_value.data()), sizeof(int) * 8, "{name} pad");

    auto {name}_paddings = op::Const("{name}_paddings")
        .set_attr_value({name}_pad_tensor);
    graph.AddOp({name}_paddings);
    auto {name}_pad = op::Pad("{name}_pad")
        .set_input_x({x_name})
        .set_input_paddings({name}_paddings);
    graph.AddOp({name}_pad);
    auto {name} = op::MaxPoolWithArgmax("{name}")
        .set_input_x({name}_pad)
        .set_attr_ksize({ksize_str})
        .set_attr_strides({strides_str})
        .set_attr_padding("VALID");
    graph.AddOp({name});
'''
        else:
            padding_str = 'VALID'
            max_pool_op = f'''
    auto {name} = op::MaxPoolWithArgmax("{name}")
        .set_input_x({x_name})
        .set_attr_ksize({ksize_str})
        .set_attr_strides({strides_str})
        .set_attr_padding("{padding_str}");
    graph.AddOp({name});
'''
        return max_pool_op

    @staticmethod
    def addmm(n: Node):
        name = n.name
        beta = 1
        alpha = 1
        if len(n.args) == 3:
            (c, a, b) = n.args
        elif len(n.args) == 4:
            (c, a, b, beta) = n.args
        elif len(n.args) == 5:
            (c, a, b, beta, alpha) = n.args 
        c_name = c.name if isinstance(c, torch.fx.node.Node) else c
        a_name = a.name if isinstance(a, torch.fx.node.Node) else a
        b_name = b.name if isinstance(b, torch.fx.node.Node) else b
        beta_str = str(beta)
        alpha_str = str(alpha)
        addmm_op = f'''
    float {name}_beta_value = {beta_str};
    float {name}_alpha_value = {alpha_str};
    auto {name}_beta_tensor = genTensor(std::vector<int64_t>(), FORMAT_ND, DT_FLOAT);
    auto {name}_alpha_tensor = genTensor(std::vector<int64_t>(), FORMAT_ND, DT_FLOAT);
    setTensorData({name}_beta_tensor, reinterpret_cast<uint8_t*>(&{name}_beta_value), sizeof(float), "{name} beta");
    setTensorData({name}_alpha_tensor, reinterpret_cast<uint8_t*>(&{name}_alpha_value), sizeof(float), "{name} alpha");

    auto {name}_beta = op::Const("{name}_beta")
        .set_attr_value({name}_beta_tensor);
    auto {name}_alpha = op::Const("{name}_alpha")
        .set_attr_value({name}_alpha_tensor);

    auto {name}_c_beta = op::Mul("{name}_c_beta")
        .set_input_x1({c_name})
        .set_input_x2({name}_beta);
    graph.AddOp({name}_c_beta);

    auto {name}_a_alpha = op::Mul("{name}_a_alpha")
        .set_input_x1({a_name})
        .set_input_x2({name}_alpha);
    graph.AddOp({name}_a_alpha);

    auto {name}_matmul = op::MatMul("{name}_matmul")
        .set_input_x1({name}_a_alpha)
        .set_input_x2({b_name});
    graph.AddOp({name}_matmul);

    auto {name} = op::AddV2("{name}")
        .set_input_x1({name}_c_beta)
        .set_input_x2({name}_matmul);
    graph.AddOp({name});
'''
        return addmm_op

    @staticmethod
    def var(n: Node):
        name = n.name
        (x, axes) = n.args
        axes = list(axes)
        correction = n.kwargs['correction']
        keep_dims = n.kwargs['keepdim']
        keep_dims_str = "true" if keep_dims else "false"
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        dims_str = '{' + ','.join(list(map(str, axes))) + '}'
        
        if correction == 1:
            unbiased = 'true'
        elif correction == 0:
            unbiased = 'false'
        else:
            raise RuntimeError("not supported yet!")
        
        var_op = f'''
    // 1. mean
    std::vector<int> {name}_axes_value {dims_str};
    std::vector<int64_t> {name}_axes_tensor_shape;
    if ({name}_axes_value.size() != 0) {{
        {name}_axes_tensor_shape.push_back({name}_axes_value.size());
    }}
    auto {name}_axes_tensor = genTensor({name}_axes_tensor_shape, FORMAT_ND, DT_INT32);
    setTensorData({name}_axes_tensor, reinterpret_cast<uint8_t*>({name}_axes_value.data()), {name}_axes_value.size() * sizeof(int), "{name}_axes");
    auto {name}_axes = op::Const("{name}_axes")
        .set_attr_value({name}_axes_tensor);
    auto {name}_mean = op::ReduceMean("{name}_mean")
        .set_input_x({x_name})
        .set_input_axes({name}_axes)
        .set_attr_keep_dims({keep_dims_str});
    graph.AddOp({name}_axes);
    graph.AddOp({name}_mean);
    
    // 2. broadcast to self
    auto {name}_input_shape = op::Shape("{name}_input_shape")
        .set_input_x({x_name});
    auto {name}_broadcast_to = op::BroadcastTo("{name}_broadcast_to")
        .set_input_x({name}_mean)
        .set_input_shape({name}_input_shape);
    graph.AddOp({name}_input_shape);
    graph.AddOp({name}_broadcast_to);
        
    // 3. ReduceStdV2Update
    auto {name} = op::ReduceStdV2Update("{name}")
        .set_input_x({x_name})
        .set_input_mean({name}_broadcast_to)
        .set_attr_dim({dims_str})
        .set_attr_unbiased({unbiased})
        .set_attr_keepdim({keep_dims_str});
    graph.AddOp({name});
'''
        return var_op

    @staticmethod
    def log(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        log_op = f'''
    auto {name} = op::Log("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return log_op

    @staticmethod
    def gather(n: Node):
        name = n.name
        (x, dim, index) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        dim_str = str(dim)
        index_name = index.name if isinstance(index, torch.fx.node.Node) else index

        gather_op = f'''
    auto {name}_dim_shape = ge::Shape({{1}});
    TensorDesc {name}_dim_desc({name}_dim_shape, FORMAT_NCHW, DT_INT32);
    Tensor {name}_dim_tensor({name}_dim_desc);
    int {name}_dim_value = {dim_str};
    setTensorData({name}_dim_tensor, reinterpret_cast<uint8_t*>(&{name}_dim_value), sizeof(int), "{name}_dim");

    auto {name}_dim = op::Const("{name}_dim")
        .set_attr_value({name}_dim_tensor);
    auto {name} = op::GatherD("{name}")
        .set_input_x({x_name})
        .set_input_dim({name}_dim)
        .set_input_index({index_name})
        .set_attr_dim({dim_str});
    graph.AddOp({name}_dim);
    graph.AddOp({name});

'''
        return gather_op

    @staticmethod
    def neg(n: Node):
        name = n.name
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x

        neg_op = f'''
    auto {name} = op::Neg("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return neg_op

    @staticmethod
    def expand(n: Node):
        name = n.name
        (x, shape) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        shape_str = '{' + ','.join(map(str, shape)) + '}'

        expand_op = f'''
    std::vector<int64_t> {name}_shape{shape_str};
    auto {name} = op::ExpandD("{name}") 
        .set_input_x({x_name})
        .set_attr_shape({name}_shape);
    graph.AddOp({name});
'''
        return expand_op

    @staticmethod
    def zeros_like(n: Node):
        name = n.name
        # TODO(tangzhiyi): ignore kwargs, need to check this
        x = n.args[0]
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x

        zeros_like_op = f'''
    auto {name} = op::ZerosLike("{name}")
        .set_input_x({x_name});
    graph.AddOp({name});
'''
        return zeros_like_op

    @staticmethod
    def scatter(n: Node):
        name = n.name
        assert len(n.args) == 4 

        (var, dim, index, value) = n.args
        var_name = var.name if isinstance(var, torch.fx.node.Node) else var
        index_name = index.name if isinstance(index, torch.fx.node.Node) else index
        dim_str = str(dim)
        if isinstance(value, torch.fx.node.Node):
            value_name = value.name
            scatter_op = f'''
    auto {name} = op::ScatterElements("{name}")
        .set_input_data({var_name})
        .set_input_indices({index_name})
        .set_input_updates({value_name})
        .set_attr_axis({dim_str});
    graph.AddOp({name});
'''
        else:
            ascend_dtype = get_ascend_dtype(n.meta['val'].dtype)
            cpp_dtype = get_cpp_dtype(n.meta['val'].dtype)
            value_str = str(value)
            scatter_op = f'''
    std::vector<int64_t> {name}_value_dims;
    TensorDesc {name}_value_desc(ge::Shape({name}_value_dims), FORMAT_NCHW, {ascend_dtype});
    Tensor {name}_value_tensor({name}_value_desc);
    {cpp_dtype} {name}_value_value = {value_str};
    setTensorData({name}_value_tensor, reinterpret_cast<uint8_t*>(&{name}_value_value), sizeof({cpp_dtype}), "{name}_value");

    auto {name}_value = op::Const("{name}_value")
        .set_attr_value({name}_value_tensor);
    auto {name}_index_shape = op::Shape("{name}_index_shape")
        .set_input_x({index_name});
    auto {name}_value_bcast = op::BroadcastTo("{name}_value_bcast")
        .set_input_x({name}_value)
        .set_input_shape({name}_index_shape);
    auto {name} = op::ScatterElements("{name}")
        .set_input_data({var_name})
        .set_input_indices({index_name})
        .set_input_updates({name}_value_bcast)
        .set_attr_axis({dim_str});
    graph.AddOp({name}_value);
    graph.AddOp({name}_index_shape);
    graph.AddOp({name}_value_bcast);
    graph.AddOp({name});
'''
        return scatter_op

    @staticmethod
    def mm(n: Node):
        name = n.name
        (x, y) = n.args
        x_name = x.name if isinstance(x, torch.fx.node.Node) else x
        y_name = y.name if isinstance(y, torch.fx.node.Node) else y

        mm_op = f'''
    auto {name} = op::MatMul("{name}")
        .set_input_x1({x_name})
        .set_input_x2({y_name});
    graph.AddOp({name});
'''
        return mm_op

    @staticmethod
    def convolution_backward(n: Node):
        name = n.name
        grad_output = n.args[0]
        input = n.args[1]
        weight = n.args[2]
        bias_size = n.args[3]
        stride = list(map(str, n.args[4]))
        padding = list(map(str, n.args[5]))
        dilation = list(map(str, n.args[6]))
        transposed = n.args[7]
        output_padding = n.args[8]
        groups = n.args[9]
        grad_input_mask = n.args[10]

        assert transposed == False
        assert output_padding == [0, 0]

        grad_out_name = grad_output.name
        input_name = input.name
        weight_name = weight.name

        conv_bwd_op = ''

        new_stride = ['1', '1', stride[0], stride[1]]
        new_padding = [padding[0], padding[0], padding[1], padding[1]]
        new_dilation = ['1', '1', dilation[0], dilation[1]]

        stride_str = '{' + ','.join(new_stride) + '}'
        padding_str = '{' + ','.join(new_padding) + '}'
        dilation_str = '{' + ','.join(new_dilation) + '}'
        groups_str = str(groups)

        # XXX(tangzhiyi): assume data format is NCHW
        data_format = 'NCHW'

        # input
        if grad_input_mask[0]:
            conv_bwd_op += f'''
    auto {name}_input_shape = op::Shape("{name}_input_shape")
        .set_input_x({input_name});
    auto {name}_input = op::Conv2DBackpropInput("{name}_input")
        .set_input_input_size({name}_input_shape)
        .set_input_filter({weight_name})
        .set_input_out_backprop({grad_out_name})
        .set_attr_strides({stride_str})
        .set_attr_pads({padding_str})
        .set_attr_dilations({dilation_str})
        .set_attr_groups({groups_str})
        .set_attr_data_format("{data_format}");
    graph.AddOp({name}_input_shape);
    graph.AddOp({name}_input);
'''

        # weight
        if grad_input_mask[1]:
            conv_bwd_op += f'''
    auto {name}_filter_shape = op::Shape("{name}_filter_shape")
        .set_input_x({weight_name});
    auto {name}_filter = op::Conv2DBackpropFilter("{name}_filter")
        .set_input_x({input_name})
        .set_input_filter_size({name}_filter_shape)
        .set_input_out_backprop({grad_out_name})
        .set_attr_strides({stride_str})
        .set_attr_pads({padding_str})
        .set_attr_dilations({dilation_str})
        .set_attr_groups({groups_str})
        .set_attr_data_format("NCHW");

    graph.AddOp({name}_filter_shape);
    graph.AddOp({name}_filter);

'''

        # TODO(tangzhiyi): bias is not supported yet
        assert grad_input_mask[2] == False

        only_input = grad_input_mask[0] == True and grad_input_mask[1] == False
        only_weight = grad_input_mask[0] == False and grad_input_mask[1] == True
        both_input_weight = grad_input_mask[0] == True and grad_input_mask[1] == True

        if only_input:
            conv_bwd_op += f'''
    auto {name} = op::IdentityN("{name}")
        .create_dynamic_input_x(2)
        .set_dynamic_input_x(0, {name}_input)
        .set_dynamic_input_x(1, {name}_input)
        .create_dynamic_output_y(2);
    graph.AddOp({name});
'''
        elif only_weight:
            conv_bwd_op += f'''
    auto {name} = op::IdentityN("{name}")
        .create_dynamic_input_x(2)
        .set_dynamic_input_x(0, {name}_filter)
        .set_dynamic_input_x(1, {name}_filter)
        .create_dynamic_output_y(2);
    graph.AddOp({name});
'''
        elif both_input_weight:
            conv_bwd_op += f'''
    auto {name} = op::IdentityN("{name}")
        .create_dynamic_input_x(2)
        .set_dynamic_input_x(0, {name}_input)
        .set_dynamic_input_x(1, {name}_filter)
        .create_dynamic_output_y(2);
    graph.AddOp({name});
'''
        else:
            raise RuntimeError('not supported!')
        return conv_bwd_op

    @staticmethod
    def max_pool2d_with_indices_backward(n: Node):
        name = n.name

        grad_output = n.args[0]
        x = n.args[1]
        kernel_size = list(map(str, n.args[2]))
        stride = list(map(str, n.args[3]))
        padding = list(map(str, n.args[4]))
        dilation = list(map(str, n.args[5]))
        ceil_mode = n.args[6]
        indices = n.args[7]

        assert len(kernel_size) == 2 or len(kernel_size) == 1
        assert len(stride) == 2 or len(stride) == 1
        assert len(padding) == 2 or len(padding) == 1
        assert len(dilation) == 2 or len(dilation) == 1

        # new_kernel_size = ['1', kernel_size[0], kernel_size[1], '1']
        # new_stride = ['1', stride[0], stride[1], '1']
        # new_padding = ['1', padding[0], padding[1], '1']
        # new_dilation = ['1', dilation[0], dilation[1], '1']
        new_kernel_size = ['1', '1', kernel_size[0], kernel_size[1]]
        new_stride = ['1', '1', stride[0], stride[1]]
        new_padding = ['1', padding[0], padding[1], '1']
        new_dilation = ['1', dilation[0], dilation[1], '1']

        kernel_size_str = '{' + ','.join(new_kernel_size) + '}'
        stride_str = '{' + ','.join(new_stride) + '}'
        padding_str = '{' + ','.join(new_padding) + '}'
        dilation_str = '{' + ','.join(new_dilation) + '}'

        x_name = x.name
        grad_output_name = grad_output.name
        indices_name = indices.name
        ceil_mode_str = 'true' if ceil_mode else 'false'

        assert dilation == ['1', '1']

        if padding != ['0', '0']:
            padding0 = padding[0]
            padding1 = padding[1]
            padding_str = f'0, 0, 0, 0, {padding0}, {padding0}, {padding1}, {padding1}'
            maxpool_bwd_op = f'''
    TensorDesc {name}_pad_desc(ge::Shape({{4, 2}}), FORMAT_NCHW, DT_INT32);
    std::vector<int> {name}_pad_value {{ {padding_str} }};
    Tensor {name}_pad_tensor({name}_pad_desc);
    setTensorData({name}_pad_tensor, reinterpret_cast<uint8_t*>({name}_pad_value.data()), sizeof(int) * 8, "{name} pad");

    auto {name}_paddings = op::Const("{name}_paddings")
        .set_attr_value({name}_pad_tensor);
    graph.AddOp({name}_paddings);
    auto {name}_pad = op::PadV3("{name}_pad")
        .set_input_x({x_name})
        .set_input_paddings({name}_paddings);
    graph.AddOp({name}_pad);
    auto {name}_fwd_out = op::MaxPool("{name}_fwd_out")
        .set_input_x({name}_pad)
        .set_attr_ksize({kernel_size_str})
        .set_attr_strides({stride_str})
        .set_attr_padding("VALID")
        .set_attr_data_format("NCHW");
    graph.AddOp({name}_fwd_out);
    
    auto {name}_bwd = op::MaxPoolGrad("{name}_bwd")
        .set_input_x1({name}_pad)
        .set_input_x2({name}_fwd_out)
        .set_input_grad({grad_output_name})
        .set_attr_ksize({kernel_size_str})
        .set_attr_strides({stride_str})
        .set_attr_padding("VALID")
        .set_attr_data_format("NCHW");
    graph.AddOp({name}_bwd);
    auto {name} = op::PadV3Grad("{name}")
        .set_input_x({name}_bwd)
        .set_input_paddings({name}_paddings);
    graph.AddOp({name});
    
'''
        else:
            maxpool_bwd_op = f'''
    auto {name}_fwd_out = op::MaxPool("{name}_fwd_out")
        .set_input_x({x_name})
        .set_attr_ksize({kernel_size_str})
        .set_attr_strides({stride_str})
        .set_attr_padding("VALID")
        .set_attr_data_format("NCHW");
    graph.AddOp({name}_fwd_out);
    auto {name} = op::MaxPoolGrad("{name}")
        .set_input_x1({x_name})
        .set_input_x2({name}_fwd_out)
        .set_input_grad({grad_output_name})
        .set_attr_ksize({kernel_size_str})
        .set_attr_strides({stride_str})
        .set_attr_padding("VALID")
        .set_attr_data_format("NCHW");
    graph.AddOp({name});
'''
        return maxpool_bwd_op

    @staticmethod
    def where(n: Node):
        name = n.name

        assert len(n.args) == 3
        (cond, x1, x2) = n.args
        cond_name = cond.name

        # TODO(tangzhiyi): need to process scalars
        assert isinstance(x1, torch.fx.node.Node)
        assert isinstance(x2, torch.fx.node.Node)

        x1_name = x1.name
        x2_name = x2.name

        where_op = f'''
    // 1. broadcast
    auto {name}_shape = op::Shape("{name}_cond_shape")
        .set_input_x({cond_name});
    auto {name}_x1_bcast = op::BroadcastTo("{name}_x1_bcast")
        .set_input_x({x1_name})
        .set_input_shape({name}_shape);
    auto {name}_x2_bcast = op::BroadcastTo("{name}_x2_bcast")
        .set_input_x({x2_name})
        .set_input_shape({name}_shape);
    auto {name} = op::Select("{name}")
        .set_input_condition({cond_name})
        .set_input_x1({name}_x1_bcast)
        .set_input_x2({name}_x2_bcast);
    graph.AddOp({name}_shape);
    graph.AddOp({name}_x1_bcast);
    graph.AddOp({name}_x2_bcast);
    graph.AddOp({name});
'''
        return where_op

    @staticmethod
    def le(n: Node):
        name = n.name
        (x1, x2) = n.args
        x1_name = x1.name
        if isinstance(x2, torch.fx.node.Node):
            x2_name = x2.name
            le_op = f'''
    auto {name} = op::LessEqual("{name}")
        .set_input_x1({x1_name})
        .set_input_x2({x2_name});
    graph.AddOp({name});
'''
        else:
            # TODO(tangzhiyi): get value type, now assume float
            x2_str = str(x2)
            le_op = f'''
    std::vector<int64_t> {name}_x2_dims;
    TensorDesc {name}_x2_desc(ge::Shape({name}_x2_dims), FORMAT_NCHW, DT_FLOAT);
    Tensor {name}_x2_tensor({name}_x2_desc);
    float {name}_x2_value = {x2_str};
    setTensorData({name}_x2_tensor, reinterpret_cast<uint8_t*>(&{name}_x2_value), sizeof(float), "{name}_x2");

    auto {name}_x2 = op::Const("{name}_x2")
        .set_attr_value({name}_x2_tensor);
    auto {name} = op::LessEqual("{name}")
        .set_input_x1({x1_name})
        .set_input_x2({name}_x2);
    graph.AddOp({name}_x2);
    graph.AddOp({name});
'''
        return le_op

    @staticmethod
    def scalar_tensor(n: Node):
        name = n.name
        val = n.args[0]
        val_str = str(val)
        torch_dtype = n.kwargs['dtype']
        cpp_dtype = get_cpp_dtype(torch_dtype)
        ascend_dtype = get_ascend_dtype(torch_dtype)
        scalar_tensor_op = f'''
    auto {name}_val_tensor = genTensor(std::vector<int64_t>(), FORMAT_NCHW, {ascend_dtype});
    {cpp_dtype} {name}_val_value = {val_str};
    setTensorData({name}_val_tensor, reinterpret_cast<uint8_t*>(&{name}_val_value), sizeof({cpp_dtype}), "{name}_val");
    auto {name} = op::Const("{name}")
        .set_attr_value({name}_val_tensor);
    graph.AddOp({name});
'''
        return scalar_tensor_op


    @staticmethod
    def ret_tuple(n: Node):
        name = n.name
        in1 = n.args[0]
        in2 = n.args[1]

        tuple_op = f'''
    auto {name} = op::IdentityN("{name}")
        .create_dynamic_input_x(2)
        .set_dynamic_input_x(0, {in1})
        .set_dynamic_input_x(1, {in2})
        .create_dynamic_output_y(2);
    graph.AddOp({name});
'''

        return tuple_op


    @staticmethod
    def ret_triple(n: Node):
        name = n.name
        in1 = n.args[0]
        in2 = n.args[1]
        in3 = n.args[2]

        triple_op = f'''
    auto {name} = op::IdentityN("{name}")
        .create_dynamic_input_x(3)
        .set_dynamic_input_x(0, {in1})
        .set_dynamic_input_x(1, {in2})
        .set_dynamic_input_x(2, {in3})
        .create_dynamic_output_y(3);
    graph.AddOp({name});
'''

        return triple_op


