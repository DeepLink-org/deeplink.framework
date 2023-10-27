import torch
import functools
from . import tops_op
from abc import ABC, abstractmethod
import numbers
import torch.fx.traceback as fx_traceback
from torch.fx import Proxy
from torch.fx.node import Argument, Target
import operator
from typing import Any, Dict, Tuple
from dicp.dynamo_bridge.op_transformer import SingleOpTransformer
from dicp.dynamo_bridge.compile_fx import is_torch_210
from typing import (
    Optional,
)
from torch.types import (
    Number,
)

conversions = {}
patterns = []
aten = torch.ops.aten
prims = torch.ops.prims

def args_kwargs_unchange(args, kwargs):
    return args, kwargs

def _register_conversion(
    aten_fn, decomp_fn, process_args_kwargs_fn=None
):
    register_op_singleton_flag = isinstance(decomp_fn, type) and issubclass(decomp_fn, tops_op.Operator)
    if register_op_singleton_flag:
        wrapped = (decomp_fn.get_singleton(),
                   args_kwargs_unchange if process_args_kwargs_fn is None else process_args_kwargs_fn)
    else:
        @functools.wraps(decomp_fn)
        def wrapped(*args, **kwargs):
            return decomp_fn(*args, **kwargs)

    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in conversions:
                    aten_fn.append(other_fn)

    conversions.update({fn: wrapped for fn in aten_fn})
    if register_op_singleton_flag:
        return wrapped[0]
    else:
        return wrapped

def register_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_conversion,
        aten_fn,
    )

class AtenToTopsTrasformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        if target in self._conversions:
            converted_target = self._conversions[target]
            if isinstance(converted_target, tuple):
                # converted_target: (Operation, process_args_kwargs_fn)
                out, process_fn = converted_target
                args, kwargs = process_fn(args, kwargs)
            else:
                out = self._conversions[target](self.get_proxy, *args, **kwargs)
            if isinstance(out, Proxy):
                out.node.meta = fx_traceback.get_current_meta()
                return out
            proxy = self.tracer.create_proxy('call_function', out, args, kwargs)
            proxy.node.meta = fx_traceback.get_current_meta()
            return proxy
        return super().call_function(target, args, kwargs)


@register_conversion(aten.add.Tensor)
def Add(get_proxy, x, y, alpha: Optional[Number] = 1):
    y_node = y.node if isinstance(y, torch.fx.proxy.Proxy) else y
    try:
        in_dtype = x.node.meta["val"].dtype
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        if in_dtype != out_dtype:
            x = get_proxy(tops_op.Convert, (x, out_dtype))
    except:
        pass
    if not isinstance(y_node, torch.fx.node.Node):
        y = y * alpha
    elif alpha != 1:
        y = get_proxy(tops_op.Mul, (y, alpha))
    return get_proxy(tops_op.Add, (x, y))

Abs = torch.fx.wrap(register_conversion(aten.abs)(tops_op.Abs))
AddDefalut = torch.fx.wrap(register_conversion(aten.add.default)(tops_op.AddDefalut))
AddScalar = torch.fx.wrap(register_conversion(aten.add.Scalar)(tops_op.AddScalar))

@register_conversion(aten.mul)
def Mul(get_proxy, a, b):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta") and 'val' in a.node.meta:
            if (a.node.meta['val'].dtype == torch.complex64) or (a.node.meta['val'].dtype == torch.cfloat) :
                return tops_op.ComplexMul(a, b)
    return tops_op.Mul(a, b)

MulScalar = torch.fx.wrap(register_conversion(aten.mul.Scalar)(tops_op.MulScalar))

@register_conversion(aten.div)
def Div(get_proxy, a, b):
    a_node = a.node if isinstance(a, torch.fx.proxy.Proxy) else a
    in_dtype = a_node.meta["val"].dtype
    out_dtype = fx_traceback.get_current_meta()['val'].dtype
    if in_dtype is torch.float16 or out_dtype is torch.float16:
        a = get_proxy(tops_op.Convert, (a, torch.float32))
        if not isinstance(b, numbers.Number):
            b = get_proxy(tops_op.Convert, (b, torch.float32))
        res = get_proxy(tops_op.Div, (a, b))
        return get_proxy(tops_op.Convert, (res, torch.float16))
    return get_proxy(tops_op.Div, (a, b))

Sub = torch.fx.wrap(register_conversion(aten.sub)(tops_op.Sub))
Sqrt = torch.fx.wrap(register_conversion(aten.sqrt)(tops_op.Sqrt))
Reciprocal = torch.fx.wrap(register_conversion(aten.reciprocal)(tops_op.Reciprocal))
Rsqrt = torch.fx.wrap(register_conversion(aten.rsqrt)(tops_op.Rsqrt))
Exp = torch.fx.wrap(register_conversion(aten.exp)(tops_op.Exp))
Relu = torch.fx.wrap(register_conversion(aten.relu)(tops_op.Relu))

@register_conversion(aten.sum)
def ReduceSum(get_proxy, a, *args, **kwargs):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta"):
            in_dtype = a.node.meta["val"].dtype
            out_dtype = fx_traceback.get_current_meta()['val'].dtype
            if in_dtype != out_dtype:
                a = get_proxy(tops_op.Convert, (a, out_dtype))
    return get_proxy(tops_op.ReduceSum, (a, *args), kwargs)

GetTupleElement = torch.fx.wrap(register_conversion(operator.getitem)(tops_op.GetTupleElement))

@register_conversion(aten.index.Tensor)
def Index(get_proxy, *args, **kwargs):
    assert len(args[1]) == 1, f"Only support aten.index with one index arg" 
    idx_rank = len(args[1][0].node.meta['val'].shape)
    slice_size = list(args[0].node.meta['val'].shape)
    slice_size[0] = 1
    return get_proxy(tops_op.XlaGather, (args[0], args[1][0], 
           [idx_rank,], [0,], [0,] , idx_rank, [1, args[0].node.meta['val'].shape[1]]))

# tops_dropout only returns a tensor, not a tuple of tensor
@register_conversion(aten.native_dropout.default)
def NativeDropout(get_proxy, *args, **kwargs):
    dropout = get_proxy(tops_op.NativeDropout, args)
    ne = get_proxy(tops_op.NotEqual, (dropout, 0))
    return get_proxy(tops_op.MakeTuple, (dropout, ne))

Squeeze = torch.fx.wrap(register_conversion(aten.squeeze)(tops_op.Squeeze))
Unsqueeze = torch.fx.wrap(register_conversion(aten.unsqueeze)(tops_op.Unsqueeze))
Permute = torch.fx.wrap(register_conversion(aten.permute)(tops_op.Transpose))
Transpose = torch.fx.wrap(register_conversion(aten.transpose)(tops_op.Transpose1))
Hardswish = torch.fx.wrap(register_conversion(aten.hardswish)(tops_op.Hardswish))
HardswishBackward = torch.fx.wrap(register_conversion(aten.hardswish_backward)(tops_op.HardswishBackward))
Clone = torch.fx.wrap(register_conversion(aten.clone)(tops_op.Clone))
Copy = torch.fx.wrap(register_conversion(aten.copy.default)(tops_op.Copy))
Copy_ = torch.fx.wrap(register_conversion(aten.copy_.default)(tops_op.Copy))
LiftFreshCopy = torch.fx.wrap(register_conversion(aten.lift_fresh_copy.default)(tops_op.LiftFreshCopy))
Alias = torch.fx.wrap(register_conversion(aten.alias)(tops_op.Alias))
Neg = torch.fx.wrap(register_conversion(aten.neg)(tops_op.Neg))

@register_conversion(aten.mean)
def ReduceMean(get_proxy, a, dim, keepdim=False, **kwargs):
    in_shape = a.node.meta["val"].shape
    dim = [(item + len(in_shape)) if item < 0 else item for item in dim]
    return get_proxy(tops_op.ReduceMean, (a, dim, keepdim))

Less = torch.fx.wrap(register_conversion(aten.lt.Tensor)(tops_op.Less))
LessEqual = torch.fx.wrap(register_conversion(aten.le.Scalar)(tops_op.LessEqual))
Equal = torch.fx.wrap(register_conversion(aten.eq.Tensor)(tops_op.Equal))
EqualScalar = torch.fx.wrap(register_conversion(aten.eq.Scalar)(tops_op.EqualScalar))

@register_conversion(aten.ne.Scalar)
def NotEqual(get_proxy, a, b):
    data_type = a.node.meta["val"].dtype
    return get_proxy(tops_op.NotEqual, (data_type, a, b))

@register_conversion(aten.view)
def Reshape(get_proxy, *args, **kwargs):
    if  args[0].node.meta["val"].dtype in (torch.cfloat, torch.cdouble):
        x = get_proxy(tops_op.GetTupleElement, (args[0], 0))
        x = get_proxy(tops_op.Reshape, (x, *args[1:]), kwargs)
        y = get_proxy(tops_op.GetTupleElement, (args[0], 1))
        y = get_proxy(tops_op.Reshape, (y, *args[1:]), kwargs)
        return get_proxy(tops_op.MakeTuple, (x, y))
    return get_proxy(tops_op.Reshape, args, kwargs)

@register_conversion(aten.convolution)
def Convolution(get_proxy, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    inputs = [item for item in (input, weight, bias) if item is not None]
    padding = [padding[0], padding[0]] if len(padding) == 1 else list(padding)
    inputs = [inputs, f"{{{', '.join(map(str,stride))}}}", f"{{{padding[0]}, {padding[0]}, {padding[1]}, {padding[1]}}}",
              f"{{{', '.join(map(str,  dilation))}}}"]
    return get_proxy(tops_op.Convolution, (inputs, input, weight, bias, stride, padding, dilation,
                                                           transposed, output_padding, groups))

@register_conversion(aten.convolution_backward.default)
def ConvolutionBackward(get_proxy, grad_output, input, weight, bias_size, stride, padding, dilation, *args, **kwargs):
    inputs = [item for item in (grad_output, input, weight)]
    inputs = [inputs, f"{{{', '.join(map(str, bias_size))}}}", f"{{{', '.join(map(str, stride))}}}",
              f"{{{', '.join(map(str, padding))}}}", f"{{{', '.join(map(str, dilation))}}}"]
    return get_proxy(tops_op.ConvolutionBackward, (inputs, grad_output, input, weight, bias_size, 
                                                                   stride, padding, dilation, *args), kwargs)

@register_conversion(aten.max_pool2d_with_indices)
def Max_pool2d_with_indices(get_proxy, x, kernel_size, stride=[], padding=[0, 0], dilation=[1, 1], ceil_mode=False):
    ksize = f"{{{', '.join(map(str, kernel_size))}}}"
    enflame_stride = f"{{{', '.join(map(str, stride))}}}" if stride else f"{{1, 1}}"
    padding = [padding[0], padding[0]] if len(padding) == 1 else list(padding)
    enflame_padding = f"{{{padding[0]}, {padding[0]}, {padding[1]}, {padding[1]}}}"
    out_shape = fx_traceback.get_current_meta()["val"][0].shape
    inputs = [ksize, enflame_stride, enflame_padding, f"{{{', '.join(map(str, out_shape))}}}"]
    return get_proxy(tops_op.Max_pool2d_with_indices, (inputs, x, kernel_size, stride, padding, dilation, ceil_mode))

MaxPool2DBackward = torch.fx.wrap(register_conversion(aten.max_pool2d_with_indices_backward)(tops_op.Max_pool2d_with_indices_backward))

@register_conversion(aten._adaptive_avg_pool2d.default)
def Adaptive_avg_pool2d(get_proxy, *args, **kwargs):
    assert len(args) == 2 and args[1] == [1, 1], "limited support"
    reudce_dim = f"{{2, 3}}"
    return get_proxy(tops_op.Adaptive_avg_pool2d, (reudce_dim, *args), kwargs)

@register_conversion(aten._adaptive_avg_pool2d_backward.default)
def Adaptive_avg_pool2d_backward(get_proxy, grad_output, input):
    out_shape = fx_traceback.get_current_meta()["val"].shape
    expand = get_proxy(tops_op.Expand, (grad_output, out_shape))
    value = out_shape[2] * out_shape[3]
    scalar = get_proxy(tops_op.Scalar, (value, ))
    return get_proxy(tops_op.Div, (expand, scalar))

Gather = torch.fx.wrap(register_conversion(aten.gather)(tops_op.Gather))
Log = torch.fx.wrap(register_conversion(aten.log)(tops_op.Log))
ReduceMax = torch.fx.wrap(register_conversion(aten.amax)(tops_op.ReduceMax))
DotGeneral = torch.fx.wrap(tops_op.DotGeneral)
BatchNorm = torch.fx.wrap(register_conversion(aten._native_batch_norm_legit_functional.default)(tops_op.BatchNorm))

@register_conversion(aten.native_batch_norm_backward.default)
def BatchNormBackward(*args, **kwargs):
    return tops_op.BatchNormBackward(*args, **kwargs)

@register_conversion(aten._softmax)
def Softmax(get_prxy, a, dim, half_to_float):
    out_shape = fx_traceback.get_current_meta()["val"].shape
    dim = dim + len(out_shape) if dim < 0 else dim
    return get_prxy(tops_op.Softmax, (a, dim, half_to_float))

Bmm = torch.fx.wrap(register_conversion(aten.bmm.default)(tops_op.Bmm))
Dot = torch.fx.wrap(register_conversion(aten.dot.default)(tops_op.Dot))

@register_conversion(aten.mm)
def Gemm(get_proxy, *args, **kwargs):
    return get_proxy(tops_op.Gemm, args)
Gemm = torch.fx.wrap(tops_op.Gemm)

@register_conversion(aten.bmm.default)
def Bmm(get_proxy, *args, **kwargs):
    return get_proxy(tops_op.DotGeneral, (*args, 
          [0,], [0,], [2,], [1,]))

@register_conversion(aten.cat.default)
def Concatenate(get_proxy, *args, **kwargs):
    tensors = []
    for arg in args[0]:
        if torch.numel(arg.node.meta['val']):
            tensors.append(arg)
    dim = 0 if len(args) < 2 else args[1] 
    dim = dim % len(args[0][0].node.meta["val"].shape)
    return get_proxy(tops_op.Concatenate, (args[0], dim))

EmptyLike = torch.fx.wrap(register_conversion(aten.empty_like.default)(tops_op.EmptyLike))
Bernoulli = torch.fx.wrap(register_conversion(aten.bernoulli.p)(tops_op.Bernoulli))
NewEmptyStrided = torch.fx.wrap(register_conversion(aten.new_empty_strided.default)(tops_op.NewEmptyStrided))
Expand = torch.fx.wrap(register_conversion(aten.expand.default)(tops_op.Expand))
Full = torch.fx.wrap(register_conversion(aten.full.default)(tops_op.Full))
FullLike = torch.fx.wrap(register_conversion(aten.full_like.default)(tops_op.FullLike))
Max = torch.fx.wrap(register_conversion(aten.maximum.default)(tops_op.Max))
Pow = torch.fx.wrap(register_conversion(aten.pow.Tensor_Scalar)(tops_op.Pow))
Sigmoid = torch.fx.wrap(register_conversion(aten.sigmoid.default)(tops_op.Sigmoid))

@register_conversion(aten.slice.Tensor)
def Slice(get_proxy, a, dim=0, start=0, end=-1, step=1, **kwargs):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta"):
            in_shape = a.node.meta["val"].shape
            out_shape = fx_traceback.get_current_meta()["val"].shape
            if in_shape != out_shape:
                start = start % in_shape[dim]
                end = end + in_shape[dim] if end < 0 else end
                end = in_shape[dim] if end > in_shape[dim] else end
                return get_proxy(tops_op.SliceInDim, (a, dim, start, end, step), kwargs)
            start_indices = f"{{{', '.join(map(str, [0] * len(out_shape)))}}}"
            limit_indices = f"{{{str(in_shape).split('[')[-1].split(']')[0]}}}"
            strides = f"{{{', '.join(map(str, [1] * len(out_shape)))}}}"
    return get_proxy(tops_op.Slice, (start_indices, limit_indices, strides, a, dim, start, end, step), kwargs)

@register_conversion(aten.slice_scatter.default)
def SliceScatter(get_proxy, a, b, dim=0, start=0, end=-1, step=1):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta"):
            operand_shape = a.node.meta["val"].shape
            end = end % operand_shape[dim] if end < operand_shape[dim] else operand_shape[dim]
            assert end == operand_shape[dim] and step == 1, "limited support"
            return get_proxy(tops_op.SliceScatter, (a, b, dim, start, end, step))

@register_conversion(aten.select.int)
def Select(get_proxy, a, dim, index):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta"):
            in_shape = a.node.meta["val"].shape
            index = index % in_shape[dim]
            slice = get_proxy(tops_op.SliceInDim, (a, dim, index, index + 1, 1))
            return get_proxy(tops_op.Squeeze, (slice, dim))

Where = torch.fx.wrap(register_conversion(aten.where.self)(tops_op.Where))
Scatter = torch.fx.wrap(register_conversion(aten.scatter.value)(tops_op.Scatter))
ZerosLike = torch.fx.wrap(register_conversion(aten.zeros_like)(tops_op.ZerosLike))
OnesLike = torch.fx.wrap(register_conversion(aten.ones_like)(tops_op.OnesLike))

@register_conversion(aten.scalar_tensor.default)
def Scalar(get_proxy, a, **kwargs):
    if "dtype" in kwargs:
        real_dtype = kwargs["dtype"]
        if not real_dtype in (torch.int64, torch.float32):
            kwargs["dtype"] = torch.float32
            scalar = get_proxy(tops_op.Scalar, (a,), kwargs)
            return get_proxy(tops_op.Convert(), (scalar, real_dtype))
    return get_proxy(tops_op.Scalar, (a,), kwargs)

@register_conversion(aten.embedding)
def Embedding(get_proxy, *args, **kwargs):
    idx_rank = len(args[1].node.meta['val'].shape)
    return get_proxy(tops_op.XlaGather, (*args, 
           [idx_rank,], [0,], [0,] , idx_rank, [1, args[0].node.meta['val'].shape[1]]))

Convert = torch.fx.wrap(register_conversion(prims.convert_element_type)(tops_op.Convert))
ViewAsComplex = torch.fx.wrap(register_conversion(aten.view_as_complex)(tops_op.ViewAsComplex))
ViewAsReal = torch.fx.wrap(register_conversion(aten.view_as_real)(tops_op.ViewAsReal))
UnsafeView = torch.fx.wrap(register_conversion(aten._unsafe_view.default)(tops_op.UnsafeView))
Logsoftmax = torch.fx.wrap(register_conversion(aten._log_softmax.default)(tops_op.Logsoftmax))
ViewAsComplex = torch.fx.wrap(register_conversion(aten.view_as_complex)(tops_op.ViewAsComplex))
ViewAsReal = torch.fx.wrap(register_conversion(aten.view_as_real)(tops_op.ViewAsReal))

@register_conversion(aten.gelu.default)
def Gelu(get_proxy, *args, **kwargs):
    approximate = 'true' if ('approximate' in kwargs 
        and kwargs["approximate"] == 'tanh') else 'false'
    return get_proxy(tops_op.Gelu, (args[0], approximate))

@register_conversion(aten.gelu_backward.default)
def gelubackward(get_proxy, *args, **kwargs):
    approximate = 'true' if ('approximate' in kwargs 
        and kwargs["approximate"] == 'tanh') else 'false'
    return get_proxy(tops_op.GeluBackward, (args[0], args[1], approximate))

@register_conversion(prims.iota.default)
def Iota(get_proxy, length, **kwargs):
    iota = get_proxy(tops_op.Iota, (length,), kwargs)
    if kwargs["start"] != 0 or kwargs["step"] != 1:
        offset = get_proxy(tops_op.Mul, (iota, kwargs["step"]))
        return get_proxy(tops_op.Add, (offset, kwargs["start"]))
    return iota 

# Patterns
def register_pattern(Pattern):
    # TODO OpOverloadPacket
    patterns.append(Pattern)
    return Pattern


class BaseReplacePattern(ABC):
    @abstractmethod
    def pattern(*args, **kwargs):
        pass

    @abstractmethod
    def replacement(*args, **kwargs):
        pass
    

@register_pattern
class ReplacePatternAddmm:
    def pattern(a, b, c):
        return torch.ops.aten.addmm.default(a, b, c)

    def replacement(a, b, c):
        
        return torch.ops.aten.add.Tensor(a, torch.ops.aten.mm(b, c))
    

# %var: [#users=2] = call_function[target=torch.ops.aten.var.correction]
#                                      (args = (%convolution_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
@register_pattern
class ReplacePatternVar:
    def pattern(a, b):
        return torch.ops.aten.var.correction(a, b, correction=0, keepdim=True)

    def replacement(inputs, dims):
        keepdim = True
        correction = 0
        denom = 64
        denom = denom - correction
        mean1 = torch.ops.aten.mean.dim(inputs, dims, keepdim)
        diffs = torch.ops.aten.square.default(torch.ops.aten.sub.Tensor(inputs, mean1))
        sum_results = torch.ops.aten.sum.dim_IntList(diffs, dims, keepdim)
        x_var = torch.ops.aten.div.Tensor(sum_results, denom)
        return x_var


# %var_mean_correction_4 : [#users=2] = call_function[target=torch.ops.aten.var_mean.correction]
#                                      (args = (%convolution_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
@register_pattern
class ReplacePatternVarMean:
    def pattern(a, b):
        return torch.ops.aten.var_mean.correction(a, b, correction=0, keepdim=True)

    def replacement(inputs, dims):
        keepdim = True
        correction = 0
        denom = 64
        denom = denom - correction
        mean1 = torch.ops.aten.mean.dim(inputs, dims, keepdim)
        diffs = torch.ops.aten.square.default(torch.ops.aten.sub.Tensor(inputs, mean1))
        sum_results = torch.ops.aten.sum.dim_IntList(diffs, dims, keepdim)
        x_var = torch.ops.aten.div.Tensor(sum_results, denom)
        return tops_op.ret_tuples(x_var, mean1)


@register_pattern
class ReplacePatternT:
    def pattern(a):
        return torch.ops.aten.t.default(a)

    def replacement(inputs):
        return torch.ops.aten.transpose(inputs, 0, 1)

@register_pattern
class ReplacePatternRsub:
    def pattern(a, b):
        return torch.ops.aten.rsub.Scalar(a, b)

    def replacement(a, b):
        return torch.ops.aten.sub.Scalar(b, a)


@register_pattern
class ReplacePatternSiLU:
    # silu(x) = x / (1+exp(-x)) = x*sigmoid(x)
    def pattern(a):
        return torch.ops.aten.silu.default(a)

    def replacement(a):
        return torch.ops.aten.mul.default(a, torch.ops.aten.sigmoid.default(a))


if is_torch_210:
    import functools
    from dicp.dynamo_bridge.op_transformer import (
        BackendPatternBase,
        PatternMatcherPass,
        register_backend_patterns,
    )

    tops_patterns = PatternMatcherPass()
    tops_patterns_cls_list = []
    register_tops_patterns = functools.partial(register_backend_patterns, tops_patterns_cls_list)

    @register_tops_patterns
    class GemmTransposeRhsPattern(BackendPatternBase):
        @staticmethod
        def pattern(reshaped_input, weight):
            transposed_weight = Permute(weight, [1, 0])
            return Gemm(reshaped_input, transposed_weight)

        @staticmethod
        def replacement(reshaped_input, weight):
            return DotGeneral(reshaped_input, weight, [], [], [1,], [1,])

    @register_tops_patterns
    class LlamaMatmulTransposePattern(BackendPatternBase):
        @staticmethod
        def pattern(xq, keys, expanded_xq_size, reshaped_xq_size, expanded_keys_size, reshaped_keys_size):
            xq_1 = Permute(xq, [0, 2, 1, 3])
            keys_1 = Permute(keys, [0, 2, 1, 3])
            keys_2 = Permute(keys_1, [0, 1, 3, 2])
            expanded_xq = Expand(xq_1, expanded_xq_size)
            reshaped_xq = Reshape(expanded_xq, reshaped_xq_size)
            expanded_keys = Expand(keys_2, expanded_keys_size)
            reshaped_keys = Reshape(expanded_keys, reshaped_keys_size)
            bmm_res = Bmm(reshaped_xq, reshaped_keys)
            return bmm_res

        @staticmethod
        def replacement(xq, keys):
            return DotGeneral(xq, keys, [0, 2], [0, 2], [3,], [3,])


