import torch
import functools
from . import tops_op
from abc import ABC, abstractmethod
import numbers
import torch.fx.traceback as fx_traceback
from torch.fx import Proxy
import operator
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

@register_conversion(torch.ops.aten.add.Tensor)
def Add(get_proxy, x, y, alpha: Optional[Number] = 1):
    y_node = y.node if isinstance(y, torch.fx.proxy.Proxy) else y
    out_dtype = fx_traceback.get_current_meta()['val'].dtype
    if not isinstance(y_node, torch.fx.node.Node):
        y = y * alpha
        if out_dtype == torch.float or out_dtype == torch.float16:
            return get_proxy(tops_op.Add.get_singleton(), (x, float(y)), {})
        else:
            raise ValueError(f"only support torch.add + alpha with float type")
    else:
        y = get_proxy(tops_op.Mul.get_singleton(), (y, alpha), {})
    return get_proxy(tops_op.Add.get_singleton(), (x, y), {})

Abs = torch.fx.wrap(register_conversion(torch.ops.aten.abs)(tops_op.Abs))
AddDefalut = torch.fx.wrap(register_conversion(torch.ops.aten.add.default)(tops_op.AddDefalut))
AddScalar = torch.fx.wrap(register_conversion(torch.ops.aten.add.Scalar)(tops_op.AddScalar))

@register_conversion(torch.ops.aten.mul)
def Mul(get_proxy, a, b):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta") and 'val' in a.node.meta:
            if (str(a.node.meta['val'].dtype) == "torch.complex64") or (str(a.node.meta['val'].dtype) == "torch.cfloat") :
                return tops_op.ComplexMul(a, b)
    return tops_op.Mul(a, b)

MulScalar = torch.fx.wrap(register_conversion(torch.ops.aten.mul.Scalar)(tops_op.MulScalar))

@register_conversion(torch.ops.aten.div)
def Div(get_proxy, a, b):
    a_node = a.node if isinstance(a, torch.fx.proxy.Proxy) else a
    in_dtype = a_node.meta["val"].dtype
    out_dtype = fx_traceback.get_current_meta()['val'].dtype
    if in_dtype is torch.float16 or out_dtype is torch.float16:
        a = get_proxy(tops_op.Convert.get_singleton(), (a, torch.float32), {})
        if not isinstance(b, numbers.Number):
            b = get_proxy(tops_op.Convert.get_singleton(), (b, torch.float32), {})
        res = get_proxy(tops_op.Div.get_singleton(), (a, b), {})
        return get_proxy(tops_op.Convert.get_singleton(), (res, torch.float16), {})
    return get_proxy(tops_op.Div.get_singleton(), (a, b), {})

Sub = torch.fx.wrap(register_conversion(torch.ops.aten.sub)(tops_op.Sub))
Sqrt = torch.fx.wrap(register_conversion(torch.ops.aten.sqrt)(tops_op.Sqrt))

@register_conversion(torch.ops.aten.square)
def Square(*args, **kwargs):
    return tops_op.Square(*args, **kwargs)

Reciprocal = torch.fx.wrap(register_conversion(torch.ops.aten.reciprocal)(tops_op.Reciprocal))
Rsqrt = torch.fx.wrap(register_conversion(torch.ops.aten.rsqrt)(tops_op.Rsqrt))
Exp = torch.fx.wrap(register_conversion(torch.ops.aten.exp)(tops_op.Exp))
Relu = torch.fx.wrap(register_conversion(torch.ops.aten.relu)(tops_op.Relu))

@register_conversion(torch.ops.aten.sum)
def Sum(*args, **kwargs):
    return tops_op.ReduceSum(*args, **kwargs)

@register_conversion(torch.ops.aten.sum.dim_IntList)
def Sumdim(*args, **kwargs):
    return tops_op.ReduceSum(*args, **kwargs)

Getitem = torch.fx.wrap(register_conversion(operator.getitem)(tops_op.Getitem))

@register_conversion(torch.ops.aten.index.Tensor)
def Index(*args, **kwargs):
    return tops_op.Index(*args, **kwargs)

@register_conversion(torch.ops.aten.native_dropout.default)
def NativeDropout(*args, **kwargs):
    return tops_op.NativeDropout(*args, **kwargs)

Squeeze = torch.fx.wrap(register_conversion(torch.ops.aten.squeeze)(tops_op.Squeeze))
Unsqueeze = torch.fx.wrap(register_conversion(torch.ops.aten.unsqueeze)(tops_op.Unsqueeze))
Permute = torch.fx.wrap(register_conversion(torch.ops.aten.permute)(tops_op.Transpose))
Transpose = torch.fx.wrap(register_conversion(torch.ops.aten.transpose)(tops_op.Transpose1))
Hardswish = torch.fx.wrap(register_conversion(torch.ops.aten.hardswish)(tops_op.Hardswish))
HardswishBackward = torch.fx.wrap(register_conversion(torch.ops.aten.hardswish_backward)(tops_op.HardswishBackward))
Clone = torch.fx.wrap(register_conversion(torch.ops.aten.clone)(tops_op.Clone))
Copy = torch.fx.wrap(register_conversion(torch.ops.aten.copy.default)(tops_op.Copy))
Copy_ = torch.fx.wrap(register_conversion(torch.ops.aten.copy_.default)(tops_op.Copy))
LiftFreshCopy = torch.fx.wrap(register_conversion(torch.ops.aten.lift_fresh_copy.default)(tops_op.LiftFreshCopy))
Alias = torch.fx.wrap(register_conversion(torch.ops.aten.alias)(tops_op.Alias))
Neg = torch.fx.wrap(register_conversion(torch.ops.aten.neg)(tops_op.Neg))

# %mean_dim : [#users=2] = call_function[target=torch.ops.aten.mean.dim]
#                          (args = (%relu_16, [-1, -2], True), kwargs = {})
@register_conversion(torch.ops.aten.mean)
def Mean(*args, **kwargs):
    return tops_op.ReduceMean(*args, **kwargs)

Reshape = torch.fx.wrap(register_conversion(torch.ops.aten.view)(tops_op.Reshape))

@register_conversion(torch.ops.aten.convolution)
def Convolution(*args, **kwargs):
    return tops_op.Convolution(*args, **kwargs)

@register_conversion(torch.ops.aten.convolution_backward.default)
def ConvolutionBackward(*args, **kwargs):
    return tops_op.ConvolutionBackward(*args, **kwargs)

Less = torch.fx.wrap(register_conversion(torch.ops.aten.lt.Tensor)(tops_op.Less))
LessEqual = torch.fx.wrap(register_conversion(torch.ops.aten.le.Scalar)(tops_op.LessEqual))
Equal = torch.fx.wrap(register_conversion(torch.ops.aten.eq.Tensor)(tops_op.Equal))
EqualScalar = torch.fx.wrap(register_conversion(torch.ops.aten.eq.Scalar)(tops_op.EqualScalar))
NotEqual = torch.fx.wrap(register_conversion(torch.ops.aten.ne.Scalar)(tops_op.NeScalar))

@register_conversion(torch.ops.aten.max_pool2d_with_indices)
def Max_pool2d_with_indices(*args, **kwargs):
    return tops_op.Max_pool2d_with_indices(*args, **kwargs)

@register_conversion(torch.ops.aten.max_pool2d_with_indices_backward)
def Max_pool2d_with_indices_backward(*args, **kwargs):
    return tops_op.Max_pool2d_with_indices_backward(*args, **kwargs)

@register_conversion(torch.ops.aten._adaptive_avg_pool2d.default)
def Adaptive_avg_pool2d(*args, **kwargs):
    return tops_op.Adaptive_avg_pool2d(*args, **kwargs)

@register_conversion(torch.ops.aten._adaptive_avg_pool2d_backward.default)
def Adaptive_avg_pool2d_backward(*args, **kwargs):
    return tops_op.Adaptive_avg_pool2d_backward(*args, **kwargs)

@register_conversion(torch.ops.aten.gather)
def Gather(*args, **kwargs):
    return tops_op.Gather(*args, **kwargs)

Log = torch.fx.wrap(register_conversion(torch.ops.aten.log)(tops_op.Log))

@register_conversion(torch.ops.aten.amax)
def Max(*args, **kwargs):
    return tops_op.ReduceMax(*args, **kwargs)

Gemm = torch.fx.wrap(register_conversion(torch.ops.aten.mm)(tops_op.Gemm))
DotGeneral = torch.fx.wrap(tops_op.DotGeneral.get_singleton())

@register_conversion(torch.ops.aten._native_batch_norm_legit_functional.default)
def Batchnorm(*args, **kwargs):
    return tops_op.BatchNorm(*args, **kwargs)

@register_conversion(torch.ops.aten.native_batch_norm_backward.default)
def BatchNormBackward(*args, **kwargs):
    return tops_op.BatchNormBackward(*args, **kwargs)

Softmax = torch.fx.wrap(register_conversion(torch.ops.aten._softmax.default)(tops_op.Softmax))

@register_conversion(torch.ops.aten.arange.start)
def Range(*args, **kwargs):
    return tops_op.Range(*args, **kwargs)

Bmm = torch.fx.wrap(register_conversion(torch.ops.aten.bmm.default)(tops_op.Bmm))
Dot = torch.fx.wrap(register_conversion(torch.ops.aten.dot.default)(tops_op.Dot))

@register_conversion(torch.ops.aten.cat.default)
def Concatenate(*args, **kwargs):
    return tops_op.Concatenate(*args, **kwargs)

EmptyLike = torch.fx.wrap(register_conversion(torch.ops.aten.empty_like.default)(tops_op.EmptyLike))
Bernoulli = torch.fx.wrap(register_conversion(torch.ops.aten.bernoulli.p)(tops_op.Bernoulli))
NewEmptyStrided = torch.fx.wrap(register_conversion(torch.ops.aten.new_empty_strided.default)(tops_op.NewEmptyStrided))
Expand = torch.fx.wrap(register_conversion(torch.ops.aten.expand.default)(tops_op.Expand))
Full = torch.fx.wrap(register_conversion(torch.ops.aten.full.default)(tops_op.Full))
FullLike = torch.fx.wrap(register_conversion(torch.ops.aten.full_like.default)(tops_op.FullLike))
Maximum = torch.fx.wrap(register_conversion(torch.ops.aten.maximum.default)(tops_op.Max))
Pow = torch.fx.wrap(register_conversion(torch.ops.aten.pow.Tensor_Scalar)(tops_op.Pow))
Sigmoid = torch.fx.wrap(register_conversion(torch.ops.aten.sigmoid.default)(tops_op.Sigmoid))

@register_conversion(torch.ops.aten.slice.Tensor)
def Slice(*args, **kwargs):
    return tops_op.Slice(*args, **kwargs)

@register_conversion(torch.ops.aten.slice_scatter.default)
def SliceScatter(*args, **kwargs):
    return tops_op.SliceScatter(*args, **kwargs)

@register_conversion(torch.ops.aten.index.Tensor)
def Index(*args, **kwargs):
    return tops_op.Index(*args, **kwargs)

Where = torch.fx.wrap(register_conversion(torch.ops.aten.where.self)(tops_op.Where))

@register_conversion(torch.ops.aten.select.int)
def Select(*args, **kwargs):
    return tops_op.Select(*args, **kwargs)

@register_conversion(torch.ops.aten.scatter.value)
def Scatter(*args, **kwargs):
    return tops_op.Scatter(*args, **kwargs)

ZerosLike = torch.fx.wrap(register_conversion(torch.ops.aten.zeros_like)(tops_op.ZerosLike))
OnesLike = torch.fx.wrap(register_conversion(torch.ops.aten.ones_like)(tops_op.OnesLike))

@register_conversion(torch.ops.aten.scalar_tensor.default)
def Scalar(get_proxy, a, **kwargs):
    if "dtype" in kwargs:
        real_dtype = kwargs["dtype"]
        if not real_dtype in (torch.int64, torch.float32):
            kwargs["dtype"] = torch.float32
            scalar = get_proxy(tops_op.Scalar.get_singleton(), (a,), kwargs)
            return get_proxy(tops_op.Convert(), (scalar, real_dtype), {})
    return get_proxy(tops_op.Scalar.get_singleton(), (a,), kwargs)

@register_conversion(torch.ops.aten.embedding)
def Embedding(*args, **kwargs):
    return tops_op.Embedding(*args, **kwargs)

@register_conversion(torch.ops.aten.repeat.default)
def Tile(*args, **kwargs):
    return tops_op.Tile(*args, **kwargs)

Convert = torch.fx.wrap(register_conversion(torch.ops.prims.convert_element_type)(tops_op.Convert))

@register_conversion(torch.ops.aten.view_as_complex)
def ViewAsComplex(x):
    return tops_op.ViewAsComplex(x)

@register_conversion(torch.ops.aten.view_as_real)
def ViewAsReal(*args, **kwargs):
    return tops_op.ViewAsReal(*args, **kwargs)

@register_conversion(torch.ops.aten._unsafe_view.default)
def UnsafeView(a, b):
    return tops_op.UnsafeView(a, b)

Logsoftmax = torch.fx.wrap(register_conversion(torch.ops.aten._log_softmax.default)(tops_op.Logsoftmax))

@register_conversion(torch.ops.aten.gelu.default)
def Gelu(get_proxy, *args, **kwargs):
    approximate = 'true' if ('approximate' in kwargs 
        and kwargs["approximate"] == 'tanh') else 'false'
    return get_proxy(tops_op.Gelu.get_singleton(), (args[0], approximate), {})

@register_conversion(torch.ops.aten.gelu_backward.default)
def gelubackward(get_proxy, *args, **kwargs):
    approximate = 'true' if ('approximate' in kwargs 
        and kwargs["approximate"] == 'tanh') else 'false'
    return get_proxy(tops_op.GeluBackward.get_singleton(), (args[0], args[1], approximate), {})

@register_conversion(torch.ops.prims.iota.default)
def Iota(get_proxy, length, **kwargs):
    iota = get_proxy(tops_op.Iota.get_singleton(), (length,), kwargs)
    if kwargs["start"] != 0 or kwargs["step"] != 1:
        offset = get_proxy(tops_op.Mul.get_singleton(), (iota, kwargs["step"]), {})
        return get_proxy(tops_op.Add.get_singleton(), (offset, kwargs["start"]), {})
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

'''
@register_pattern
class ReplacePatternAddAlpha:
    def pattern(a, b, c):
        return torch.ops.aten.add.Tensor(a, b, alpha = c)

    def replacement(a, b, c):
        return torch.ops.aten.add.Tensor(a, torch.ops.aten.mul.default(b, c))
'''

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
            return DotGeneral(reshaped_input, weight, "{}, {}, {1}, {1}")

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
            return DotGeneral(xq, keys, "{0, 2}, {0, 2}, {3}, {3}")


