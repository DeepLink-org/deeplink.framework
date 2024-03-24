import torch
import torch.fx
import functools
import torch.fx.traceback as fx_traceback
import os
from dicp.dynamo_bridge.graph import GraphTransformer
from dicp.vendor.TopsGraph import tops_op as tops
from torch.fx.node import Argument, Target
from torch.fx.proxy import Proxy
from typing import Any, Dict, Tuple
from torch._subclasses import FakeTensor
from . import tops_op

memory_format_conversios = {}


def args_kwargs_unchange(args, kwargs):
    return args, kwargs


def _register_conversion(tops_fn, decomp_fn, process_args_kwargs_fn=None):
    register_op_singleton_flag = isinstance(
        decomp_fn, type) and issubclass(decomp_fn, tops_op.Operator)
    if register_op_singleton_flag:
        wrapped = (decomp_fn.get_singleton(),
                   args_kwargs_unchange if process_args_kwargs_fn is None else process_args_kwargs_fn)
    else:
        @functools.wraps(decomp_fn)
        def wrapped(*args, **kwargs):
            return decomp_fn(*args, **kwargs)

    if not isinstance(tops_fn, (list, tuple)):
        tops_fn = [tops_fn]
    else:
        tops_fn = list(tops_fn)

    memory_format_conversios.update({fn: wrapped for fn in tops_fn})
    if register_op_singleton_flag:
        return wrapped[0]
    else:
        return wrapped


def register_conversion(tops_fn):
    return functools.partial(
        _register_conversion,
        tops_fn,
    )


class ConvolutionTransofrmer(torch.fx.Transformer):
    def __init__(self, graph_module):
        super().__init__(graph_module)

    def is_clast_weight(self, node):
        shape, stride = node.meta["val"].shape, node.meta["val"].stride()
        memory_format = node.meta["tensor_meta"].memory_format
        # The memory_format of weight in convolution is None after converted to channesl last manually.
        if len(shape) == 4 and memory_format is None:
            return [stride[2], stride[3], stride[1], stride[0]] == list(sorted(stride, reverse=True))
        return False

    def get_contiguous_shape(self, node_shape, node_stride):
        sorted_index = sorted(range(len(node_stride)), key=lambda i: node_stride[i], reverse=True)
        contiguous_shape = torch.Size(node_shape[i] for i in sorted_index)
        return contiguous_shape

    def set_meta_val(self, proxy):
        if "val" in proxy.node.meta and isinstance(proxy.node.meta["val"], FakeTensor):
            if self.is_clast_weight(proxy.node):
                self.clast_nodes.append(proxy.node)
            contiguous_shape = self.get_contiguous_shape(proxy.node.meta["val"].shape,
                                                         proxy.node.meta["val"].stride())
            dtype, device = proxy.node.meta["val"].dtype, proxy.node.meta["val"].device
            with proxy.node.meta["val"].fake_mode:
                    fake_value = torch.empty(contiguous_shape, dtype=dtype, device=device)
            proxy.node.meta["val"] = fake_value

    def get_proxy(self, target, args: Tuple[Argument, ...], kwargs: Dict[str, Any] = {}):
        proxy = self.tracer.create_proxy(
            'call_function', target.get_singleton(), args, kwargs)
        return proxy

    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Proxy:
        proxy = super().placeholder(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        return proxy

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if type(target) is tops.Convolution:
            x, weight = args[0], args[1]
            weight = self.get_proxy(tops.Transpose, (weight, (2, 3, 1, 0)))
            x = self.get_proxy(tops.Transpose, (x, (0, 2, 3, 1)))
            new_args = (x, weight, *args[2:], True)
            conv = self.get_proxy(tops.Convolution, new_args)
            return self.get_proxy(tops.Transpose, (conv, (0, 3, 1, 2)))
        proxy = super().call_function(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        self.set_meta_val(proxy)
        return proxy


class ChannelsLastTransformer(ConvolutionTransofrmer, torch.fx.Transformer):
    def __init__(self, graph_module):
        super().__init__(graph_module)
        self.nodes = list(graph_module.graph.nodes)
        self.call_function_nodes = [node for node in graph_module.graph.nodes \
                                    if node.op == "call_function"]
        self.call_function_index = -1
        self.clast_nodes = []
        self._conversions = memory_format_conversios
        self.dim_mapping = [0, 1, 2, 3]
        self.conv_input_index = -1
        self.curr_node = None
        self.channel_value = -1
        self.weight_transpose = False

    def is_transpose_to_clast(self, target, args):
        return target.__class__ == tops.Transpose and args[1] == (0, 2, 3, 1)

    def is_transpose_to_normal(self, target, args):
        return target.__class__ == tops.Transpose and args[1] == (0, 3, 1, 2)

    def from_conv2dclast(self, target, args):
        return isinstance(args[0], Proxy) and "convolution" in args[0].node.name
    
    def is_conv_input(self, curr_index: int, node: torch.fx.Node, target: Target, args: Tuple[Argument, ...]):
        if self.is_transpose_to_clast(target, args):
            return curr_index < len(self.call_function_nodes) and \
                self.call_function_nodes[curr_index + 1].args[0] == self.call_function_nodes[curr_index]

    def is_conv_output(self, node: torch.fx.Node, target: Target, args: Tuple[Argument, ...]):
        if self.is_transpose_to_normal(target, args):
            return node.prev == node.args[0]

    def find_next_input(self, curr_index: int, node: torch.fx.Node):
        for i in range(curr_index + 1, len(self.call_function_nodes) - 1):
            if self.call_function_nodes[i].target.__class__ == tops.Transpose and \
                    self.call_function_nodes[i].args[1] == (0, 2, 3, 1) and \
                    self.call_function_nodes[i + 1].args[0] == self.call_function_nodes[i]:
                return i
        return -1

    def make_proxy(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        if target.__class__ in self._conversions:
            converted_target = self._conversions[target.__class__]
            out = self._conversions[target.__class__](self, *args, **kwargs)
            if isinstance(out, Proxy):
                out.node.meta = fx_traceback.get_current_meta()
                self.set_meta_val(out)
                return out
            proxy = self.tracer.create_proxy('call_function', out, args, kwargs)
            proxy.node.meta = fx_traceback.get_current_meta()
            self.set_meta_val(proxy)
            return proxy
        proxy = super().call_function(target, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        self.set_meta_val(proxy)
        return proxy

    def get_proxy_store_node(self, target, args: Tuple[Argument, ...], kwargs: Dict[str, Any] = {}):
        proxy = self.get_proxy(target, args, kwargs)
        self.clast_nodes.append(proxy.node)
        return proxy

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        self.call_function_index += 1
        self.curr_node = self.call_function_nodes[self.call_function_index]
        if "convolution" in self.curr_node.name:
            self.dim_mapping = [0, 3, 1, 2]
            self.channel_value = self.curr_node.meta["val"].shape[-1]
        elif "transpose" in self.curr_node.name and \
                self.call_function_index < len(self.call_function_nodes) - 3 and \
                "convolution" in self.call_function_nodes[self.call_function_index + 3].name:
            return args[0]
        elif "transpose" in self.curr_node.name and \
                self.call_function_index < len(self.call_function_nodes) - 2 and \
                "convolution" in self.call_function_nodes[self.call_function_index + 2].name:
            return args[0]

        if self.is_conv_output(self.curr_node, target, args):
            self.conv_input_index = self.find_next_input(self.call_function_index, self.curr_node)
            self.channel_value = self.curr_node.prev.meta["val"].shape[-1]
            if self.conv_input_index != -1:
                self.clast_nodes.append(args[0].node)
                return args[0]
            else:
                proxy = super().call_function(target, args, kwargs)
                proxy.node.meta = fx_traceback.get_current_meta()
                self.set_meta_val(proxy)
                return proxy
        elif self.conv_input_index != -1:
            if self.is_conv_input(self.call_function_index, self.curr_node, target, args):
                self.channel_value = -1
                self.conv_input_index = -1
                self.clast_nodes.append(args[0].node)
                return args[0]
            else:
                proxy = self.make_proxy(target, args, kwargs)
                return proxy
        else:
            if "convolution" in self.curr_node.name:
                proxy = self.make_proxy(target, args, kwargs)
                self.clast_nodes.append(proxy.node)
            else:
                proxy = super().call_function(target, args, kwargs)
                proxy.node.meta = fx_traceback.get_current_meta()
                self.set_meta_val(proxy)
            return proxy

    @register_conversion(tops.Add)
    def Add(self, x, y):
        if x.node in self.clast_nodes and (isinstance(y, (int, float)) or y.node in self.clast_nodes):
            return self.get_proxy_store_node(tops_op.Add, (x, y))
        elif x.node in self.clast_nodes or (isinstance(y, Proxy) and y.node in self.clast_nodes):
            if not x.node in self.clast_nodes:
                x = self.get_proxy(tops_op.Transpose, (x, (0, 2, 3, 1)))
                self.clast_nodes.append(x.node)
            if not y.node in self.clast_nodes:
                y = self.get_proxy(tops_op.Transpose, (y, (0, 2, 3, 1)))
                self.clast_nodes.append(y.node)
            return self.get_proxy_store_node(tops_op.Add, (x, y))
        else:
            return self.get_proxy(tops_op.Add, (x, y))

    @register_conversion(tops.Sub)
    def Sub(self, x, y):
        if x.node in self.clast_nodes and (isinstance(y, (int, float)) or y.node in self.clast_nodes):
            return self.get_proxy_store_node(tops_op.Sub, (x, y))
        elif x.node in self.clast_nodes or (isinstance(y, Proxy) and y.node in self.clast_nodes):
            if not x.node in self. clast_nodes:
                x = self.get_proxy(tops_op.Transpose, (x, (0, 2, 3, 1)))
                self.clast_nodes.append(x.node)
            if not y.node in self.clast_nodes:
                y = self.get_proxy(tops_op.Transpose, (y, (0, 2, 3, 1)))
                self.clast_nodes.append(y.node)
            return self.get_proxy_store_node(tops_op.Sub, (x, y))
        else:
            return self.get_proxy(tops_op.Sub, (x, y))

    @register_conversion(tops.Mul)
    def Mul(self, x, y):
        if x.node in self.clast_nodes and (isinstance(y, (int, float)) or y.node in self.clast_nodes):
            return self.get_proxy_store_node(tops_op.Mul, (x, y))
        elif x.node in self.clast_nodes or (isinstance(y, Proxy) and y.node in self.clast_nodes):
            if not x.node in self. clast_nodes:
                x = self.get_proxy(tops_op.Transpose, (x, (0, 2, 3, 1)))
                self.clast_nodes.append(x.node)
            if not y.node in self.clast_nodes:
                y = self.get_proxy(tops_op.Transpose, (y, (0, 2, 3, 1)))
                self.clast_nodes.append(y.node)
            return self.get_proxy_store_node(tops_op.Mul, (x, y))
        else:
            return self.get_proxy(tops_op.Mul, (x, y))

    @register_conversion(tops.Div)
    def Div(self, x, y):
        if x.node in self.clast_nodes and (isinstance(y, (int, float)) or y.node in self.clast_nodes):
            return self.get_proxy_store_node(tops_op.Div, (x, y))
        elif x.node in self.clast_nodes or (isinstance(y, Proxy) and y.node in self.clast_nodes):
            if not x.node in self. clast_nodes:
                x = self.get_proxy(tops_op.Transpose, (x, (0, 2, 3, 1)))
                self.clast_nodes.append(x.node)
            if not y.node in self.clast_nodes:
                y = self.get_proxy(tops_op.Transpose, (y, (0, 2, 3, 1)))
                self.clast_nodes.append(y.node)
            return self.get_proxy_store_node(tops_op.Div, (x, y))
        else:
            return self.get_proxy(tops_op.Div, (x, y))

    @register_conversion(tops.Abs)
    def Abs(self, x):
        if x.node in self.clast_nodes:
            return self.get_proxy_stre_node(tops_op.Abs, (x,))
        return self.get_proxy(tops_op.Abs, (x,))

    @register_conversion(tops.Square)
    def Square(self, x):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Square, (x,))
        return self.get_proxy(tops.Square, (x,))

    @register_conversion(tops.GetTupleElement)
    def GetTupleElement(self, *args, **kwargs):
        if args[0].node.args[args[1]] in self.clast_nodes or args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.GetTupleElement, args, kwargs)
        return self.get_proxy(tops.GetTupleElement, args, kwargs)

    @register_conversion(tops.Dot)
    def Dot(self, x, y):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Dot, (x, y))
        return self.get_proxy(tops.Dot, (x, y))

    @register_conversion(tops.Expand)
    def Expand(self, *args, **kwargs):
        if (isinstance(args[0], tuple) and args[0][0].node in self.clast_nodes) or \
                args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Expand, args, kwargs)
        return self.get_proxy(tops.Expand, args, kwargs)
 
    @register_conversion(tops.Softmax)
    def Softmax(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops_op.Softmax, args, kwargs)
        return self.get_proxy(tops_op.Softmax, args, kwargs)

    @register_conversion(tops.ArgMax)
    def ArgMax(self, x, dim=0, keepdim=False, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.ArgMax, (x, dim, keepdim))
        return self.get_proxy(tops_op.ArgMax, (x, dim, keepdim))

    @register_conversion(tops.ArgMin)
    def ArgMin(self, x, dim=0, keepdim=False, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.ArgMin, (x, dim, keepdim))
        return self.get_proxy(tops_op.ArgMin, (x, dim, keepdim))

    @register_conversion(tops.SliceInDim)
    def SliceInDim(self, x, dim, start, end, step, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.SliceInDim, (x, dim, start, end, step), kwargs)
        return self.get_proxy(tops_op.SliceInDim, (x, dim, start, end, step), kwargs)

    @register_conversion(tops.Slice)
    def Slice(self, start_indices, limit_indices, strides, x, dim, start, end, step, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.Slice, (start_indices, limit_indices, strides,
                                              x, dim, start, end, step), kwargs)
        return self.get_proxy(tops_op.Slice, (start_indices, limit_indices, strides,
                                              x, dim, start, end, step), kwargs)

    @register_conversion(tops.SliceScatter)
    def SliceScatter(self, x, y, dim, start, end, step, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.SliceScatter, (x, y, dim, start, end, step), kwargs)
        return self.get_proxy(tops_op.SliceScatter, (x, y, dim, start, end, step), kwargs)

    @register_conversion(tops.ReduceSum)
    def ReduceSum(self, x, dims, keepdim, **kwargs):
        if x.node in self.clast_nodes:
            dims = sorted([self.dim_mapping[dim] for dim in dims])
            return self.get_proxy_store_node(tops_op.ReduceSum, (x, dims, keepdim), kwargs)
        return self.get_proxy(tops_op.ReduceSum, (x, dims, keepdim), kwargs)

    @register_conversion(tops.ReduceMean)
    def ReduceMean(self, x, dims, keepdim, **kwargs):
        if x.node in self.clast_nodes:
            dims = sorted([self.dim_mapping[dim] for dim in dims])
            return self.get_proxy_store_node(tops_op.ReduceMean, (x, dims, keepdim), kwargs)
        return self.get_proxy(tops_op.ReduceMean, (x, dims, keepdim), kwargs)

    @register_conversion(tops.ReduceMax)
    def ReduceMax(self, x, dims, keepdim, **kwargs):
        if x.node in self.clast_nodes:
            dims = sorted([self.dim_mapping[dim] for dim in dims])
            return self.get_proxy_store_node(tops_op.ReduceMax, (x, dims, keepdim), kwargs)
        return self.get_proxy(tops_op.ReduceMax, (x, dims, keepdim), kwargs)

    @register_conversion(tops.Squeeze)
    def Squeeze(self, x, dim, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.Squeeze, (x, dim))
        return self.get_proxy(tops_op.Squeeze, (x, dim))

    @register_conversion(tops.Unsqueeze)
    def Unsqueeze(self, x, dim, **kwargs):
        if x.node in self.clast_nodes:
            dim = self.dim_mapping[dim]
            return self.get_proxy_store_node(tops_op.Unsqueeze, (x, dim))
        return self.get_proxy(tops_op.Unsqueeze, (x, dim))

    @register_conversion(tops.Transpose)
    def Transpose(self, x, dims, **kwargs):
        if x.node in self.clast_nodes:
            dims = [self.dim_mapping[dim] for dim in dims]
            return self.get_proxy_store_node(tops_op.Transpose, (x, dims), kwargs)
        return self.get_proxy(tops_op.Transpose, (x, dims), kwargs)

    @register_conversion(tops.Reshape)
    def Reshape(self, x, new_shape, **kwargs):
        if x.node in self.clast_nodes:
            if self.channel_value != -1 and len(new_shape) == 4:
                if new_shape[1] == self.channel_value:
                    new_shape = [new_shape[0], new_shape[2], new_shape[3], new_shape[1]]
                    self.dim_mapping = [0, 3, 1, 2]
                elif new_shape[1] * new_shape[2] == self.channel_value:
                    new_shape = [new_shape[0], new_shape[3], new_shape[1], new_shape[2]]
                    self.dim_mapping = [0, 2, 3, 1]
            return self.get_proxy_store_node(tops_op.Reshape, (x, new_shape), kwargs)
        return self.get_proxy(tops_op.Reshape, (x, new_shape), kwargs)

    @register_conversion(tops.Concatenate)
    def Concatenate(self, *args):
        if args[0][0].node in self.clast_nodes:
            dim = self.dim_mapping[args[1]]
            if "convolution" in args[0][0].node.name and "convolution" in args[0][1].node.name:
                self.channel_value = args[0][0].node.meta["val"].shape[dim] + args[0][1].node.meta["val"].shape[dim]
            elif "convolution" in args[0][0].node.name:
                self.channel_value = args[0][0].node.meta["val"].shape[dim] + args[0][1].node.meta["val"].shape[args[1]]
            elif "convolution" in args[0][1].node.name:
                self.channel_value = args[0][0].node.meta["val"].shape[args[1]] + args[0][1].node.meta["val"].shape[dim]
            else:
                if self.channel_value == args[0][1].node.meta["val"].shape[args[1]]:
                    self.channel_value = args[0][0].node.meta["val"].shape[args[1]] + args[0][1].node.meta["val"].shape[args[1]]
            return self.get_proxy_store_node(tops_op.Concatenate, (args[0], dim))
        return self.get_proxy(tops_op.Concatenate, args)

    @register_conversion(tops.Clone)
    def Clone(self, x, **kwargs):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Clone, (x,))
        return self.get_proxy(tops_op.Clone, (x,))

    @register_conversion(tops_op.Cos)
    def Cos(self, x):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Cos, (x,))
        return self.get_proxy(tops.Cos, (x,))

    @register_conversion(tops.Exp)
    def Exp(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Exp, args, kwargs)
        return self.get_proxy(tops.Exp, args, kwargs)

    @register_conversion(tops.Gelu)
    def Gelu(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Gelu, args, kwargs)
        return self.get_proxy(tops.Gelu, args, kwargs)

    @register_conversion(tops.Less)
    def Less(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Less, args, kwargs)
        return self.get_proxy(tops.Less, args, kwargs)

    @register_conversion(tops.Rsqrt)
    def Rsqrt(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Rsqrt, args, kwargs)
        return self.get_proxy(tops.Rsqrt, args, kwargs)

    @register_conversion(tops.Sin)
    def Sin(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Sin, args, kwargs)
        return self.get_proxy(tops.Sin, args, kwargs)

    @register_conversion(tops.Sigmoid)
    def Sigmoid(self, *args, **kwargs):
        if args[0].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Sigmoid, args, kwargs)
        return self.get_proxy(tops.Sigmoid, args, kwargs)

    @register_conversion(tops.Where)
    def Where(self, *args, **kwargs):
        if args[1].node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Where, args, kwargs)
        return self.get_proxy(tops.Where, args, kwargs)

    @register_conversion(tops.Convert)
    def Convert(self, x, dtype):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Convert, (x, dtype))
        return self.get_proxy(tops.Convert, (x, dtype))

    @register_conversion(tops.Convolution)
    def Convolution(self, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups, is_clast=False):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.Convolution, (x, weight, bias, stride, padding, dilation,
                                                                transposed, output_padding, groups, is_clast))
        return self.get_proxy(tops.Convolution, (x, weight, bias, stride, padding, dilation,
                                                 transposed, output_padding, groups, is_clast))

    @register_conversion(tops.XlaGather)
    def XlaGather(self, operand, indices, offset_dims, collapsed_slice_dims,
                  start_index_map, index_vector_dim, slice_size, out_shape):
        if operand.node in self.clast_nodes:
            operand = self.get_proxy(tops.Transpose, (operand, (0, 3, 1, 2)))
            xla_gather = self.get_proxy(tops.XlaGather, (operand, indices, offset_dims,
                                                         collapsed_slice_dims, start_index_map,
                                                         index_vector_dim, slice_size, out_shape))
            return self.get_proxy_store_node(tops.Transpose, (xla_gather, (0, 2, 3, 1)))
        return self.get_proxy(tops.XlaGather, (operand, indices, offset_dims,
                                               collapsed_slice_dims, start_index_map,
                                               index_vector_dim, slice_size, out_shape))

    @register_conversion(tops.GroupNorm)
    def GroupNorm(self, x, weight, bias, n, c, hw, group, eps):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.GroupNorm, (x, weight, bias, n, c, hw, group, eps, True))
        return self.get_proxy(tops.GroupNorm, (x, weight, bias, n, c, hw, group, eps))

    @register_conversion(tops.LayerNorm)
    def LayerNorm(self, x, normalized_shape, weight, bias, eps):
        return self.get_proxy(tops.LayerNorm, (x, normalized_shape, weight, bias, eps, True))

    @register_conversion(tops.UpsampleNearest2d)
    def UpsampleNearest2d(self, x, output_size, scales_h=None, scales_w=None):
        if x.node in self.clast_nodes:
            return self.get_proxy_store_node(tops.UpsampleNearest2d, (x, output_size, scales_h, scales_w, True))
        return self.get_proxy(tops.UpsampleNearest2d, (x, output_size, scales_h, scales_w))


class TopsMemoryFormatTransformer():
    def transform(self, gm: torch.fx.GraphModule):
        if os.getenv("DICP_SD_CLAST", default="False") == "True":
            gm = ConvolutionTransofrmer(gm).transform()
            GraphTransformer(gm, "topsgraph").infer_shape_dtype()
            gm = ChannelsLastTransformer(gm).transform()
            GraphTransformer(gm, "topsgraph").infer_shape_dtype()
        return gm
