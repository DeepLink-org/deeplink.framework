import torch
import functools
from . import tops_op
from abc import ABC, abstractmethod
from torch.fx import Proxy
import operator

conversions = {}
patterns = []
aten = torch.ops.aten
prims = torch.ops.prims

def _register_conversion(
    aten_fn, decomp_fn
):
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
def Add(a, b, **kwargs):
    return tops_op.Add(a, b, **kwargs)

@register_conversion(torch.ops.aten.add.default)
def AddDefalut(a, b):
    return tops_op.AddDefalut(a, b)

@register_conversion(torch.ops.aten.add.Scalar)
def AddScalar(a, b):
    return tops_op.AddScalar(a, b)

@register_conversion(torch.ops.aten.abs)
def Abs(a):
    return tops_op.Abs(a)

@register_conversion(torch.ops.aten.mul)
def Mul(a, b):
    if isinstance(a, Proxy):
        if hasattr(a.node, "meta") and 'val' in a.node.meta:
            if (str(a.node.meta['val'].dtype) == "torch.complex64") or (str(a.node.meta['val'].dtype) == "torch.cfloat") :
                return tops_op.ComplexMul(a, b)
    return tops_op.Mul(a, b)

@register_conversion(torch.ops.aten.mul.Scalar)
def MulScalar(a, b):
    return tops_op.MulScalar(a, b)

@register_conversion(torch.ops.aten.div.Tensor)
def Div(a, b):
    return tops_op.Div(a, b)

@register_conversion(torch.ops.aten.div.Scalar)
def DivScalar(a, b):
    return tops_op.DivScalar(a, b)

@register_conversion(torch.ops.aten.sub)
def Sub(a, b):
    return tops_op.Sub(a, b)

@register_conversion(torch.ops.aten.sqrt)
def Sqrt(a):
    return tops_op.Sqrt(a)

@register_conversion(torch.ops.aten.square)
def Square(*args):
    return tops_op.Square(*args)

@register_conversion(torch.ops.aten.reciprocal)
def Reciprocal(a):
    return tops_op.Reciprocal(a)

@register_conversion(torch.ops.aten.rsqrt)
def Rsqrt(a):
    return tops_op.Rsqrt(a)

@register_conversion(torch.ops.aten.exp)
def Exp(a):
    return tops_op.Exp(a)

@register_conversion(torch.ops.aten.relu)
def Relu(a):
    return tops_op.Relu(a)

@register_conversion(torch.ops.aten.sum)
def Sum(*args):
    return tops_op.ReduceSum(*args)

@register_conversion(torch.ops.aten.sum.dim_IntList)
def Sumdim(*args):
    return tops_op.ReduceSum(*args)

@register_conversion(operator.getitem)
def Getitem(x, idx):
    return tops_op.Getitem(x, idx)

@register_conversion(torch.ops.aten.index.Tensor)
def Index(*args, **kwargs):
    return tops_op.Index(*args, **kwargs)

@register_conversion(torch.ops.aten.native_dropout.default)
def NativeDropout(*args, **kwargs):
    return tops_op.NativeDropout(*args, **kwargs)

# torch.ops.aten.squeeze.dim(,[])
# torch.ops.aten.squeeze.dims(,)
@register_conversion(torch.ops.aten.squeeze)
def Squeeze(a, b):
    return tops_op.Squeeze(a, b)

@register_conversion(torch.ops.aten.unsqueeze)
def Unsqueeze(a, b):
    return tops_op.Unsqueeze(a, b)

@register_conversion(torch.ops.aten.permute)
def Permute(a, b):
    return tops_op.Transpose(a, b)

@register_conversion(torch.ops.aten.transpose)
def Transpose(a, b, c):
    return tops_op.Transpose1(a, b, c)

@register_conversion(torch.ops.aten.hardswish)
def Hardswish(a):
    return tops_op.Hardswish(a)

@register_conversion(torch.ops.aten.hardswish_backward)
def hardswishbackward(a, b):
    return tops_op.HardswishBackward(a, b)

@register_conversion(torch.ops.aten.clone)
def Clone(*args, **kargs):
    return tops_op.Clone(*args, **kargs)

@register_conversion(torch.ops.aten.copy.default)
def Copy(*args, **kwargs):
    return tops_op.Copy(*args, **kwargs)

@register_conversion(torch.ops.aten.lift_fresh_copy.default)
def LiftFreshCopy(*args, **kwargs):
    return tops_op.LiftFreshCopy(*args, **kwargs)

Alias = register_conversion(torch.ops.aten.alias)(tops_op.Alias)
torch.fx.wrap("Alias")

@register_conversion(torch.ops.aten.neg)
def Neg(*args):
    return tops_op.Neg(*args)

# %mean_dim : [#users=2] = call_function[target=torch.ops.aten.mean.dim]
#                          (args = (%relu_16, [-1, -2], True), kwargs = {})
@register_conversion(torch.ops.aten.mean)
def Mean(*args):
    return tops_op.ReduceMean(*args)

@register_conversion(torch.ops.aten.view)
def View(a, b):
    return tops_op.Reshape(a, b)

@register_conversion(torch.ops.aten.convolution)
def Convolution(*args):
    return tops_op.Convolution(*args)

@register_conversion(torch.ops.aten.convolution_backward.default)
def ConvolutionBackward(*args):
    return tops_op.ConvolutionBackward(*args)

@register_conversion(torch.ops.aten.lt.Tensor)
def LtTensor(*args):
    return tops_op.LtTensor(*args)

@register_conversion(torch.ops.aten.le.Scalar)
def Le(*args):
    return tops_op.LessEqual(*args)

@register_conversion(torch.ops.aten.ne.Scalar)
def NeScalar(*args):
    return tops_op.NeScalar(*args)

@register_conversion(torch.ops.aten.max_pool2d_with_indices)
def Max_pool2d_with_indices(*args):
    return tops_op.Max_pool2d_with_indices(*args)

@register_conversion(torch.ops.aten.max_pool2d_with_indices_backward)
def Max_pool2d_with_indices_backward(*args):
    return tops_op.Max_pool2d_with_indices_backward(*args)

@register_conversion(torch.ops.aten._adaptive_avg_pool2d.default)
def Adaptive_avg_pool2d(*args, **kwargs):
    return tops_op.Adaptive_avg_pool2d(*args, **kwargs)

@register_conversion(torch.ops.aten._adaptive_avg_pool2d_backward.default)
def Adaptive_avg_pool2d_backward(*args, **kwargs):
    return tops_op.Adaptive_avg_pool2d_backward(*args, **kwargs)

@register_conversion(torch.ops.aten.gather)
def Gather(*args):
    return tops_op.Gather(*args)

@register_conversion(torch.ops.aten.log)
def Log(*args):
    return tops_op.Log(*args)

@register_conversion(torch.ops.aten.amax)
def Max(*args, **kwargs):
    return tops_op.ReduceMax(*args, **kwargs)

@register_conversion(torch.ops.aten.mm)
def Gemm(*args, **kwargs):
    return tops_op.Gemm(*args, **kwargs)

@register_conversion(torch.ops.aten._native_batch_norm_legit_functional.default)
def Batchnorm(*args, **kwargs):
    return tops_op.BatchNorm(*args, **kwargs)

@register_conversion(torch.ops.aten.native_batch_norm_backward.default)
def BatchNormBackward(*args, **kwargs):
    return tops_op.BatchNormBackward(*args, **kwargs)

@register_conversion(torch.ops.aten._softmax.default)
def Softmax(*args, **kwargs):
    return tops_op.Softmax(*args, **kwargs)

@register_conversion(torch.ops.aten.arange.start)
def Range(*args, **kwargs):
    return tops_op.Range(*args, **kwargs)

@register_conversion(torch.ops.aten.bmm.default)
def Dotgeneral(*args, **kwargs):
    return tops_op.Dotgeneral(*args, **kwargs)

@register_conversion(torch.ops.aten.dot.default)
def Dot(*args, **kwargs):
    return tops_op.Dot(*args, **kwargs)

@register_conversion(torch.ops.aten.cat.default)
def Concatenate(*args, **kwargs):
    return tops_op.Concatenate(*args, **kwargs)

@register_conversion(torch.ops.aten.empty_like.default)
def EmptyLike(*args, **kwargs):
    return tops_op.EmptyLike(*args, **kwargs)

@register_conversion(torch.ops.aten.new_empty_strided.default)
def NewEmptyStrided(*args, **kwargs):
    return tops_op.NewEmptyStrided(*args, **kwargs)

@register_conversion(torch.ops.aten.eq.Tensor)
def Euqal(*args, **kwargs):
    return tops_op.Euqal(*args, **kwargs)

@register_conversion(torch.ops.aten.expand.default)
def Expand(*args, **kwargs):
    return tops_op.Expand(*args, **kwargs)

@register_conversion(torch.ops.aten.full.default)
def Full(*args, **kwargs):
    return tops_op.Full(*args, **kwargs)

@register_conversion(torch.ops.aten.full_like.default)
def FullLike(*args, **kwargs):
    return tops_op.FullLike(*args, **kwargs)

@register_conversion(torch.ops.aten.maximum.default)
def Maximum(*args, **kwargs):
    return tops_op.Max(*args, **kwargs)

@register_conversion(torch.ops.aten.pow.Tensor_Scalar)
def Pow(*args, **kwargs):
    return tops_op.Pow(*args, **kwargs)

@register_conversion(torch.ops.aten.sigmoid.default)
def Sigmoid(*args, **kwargs):
    return tops_op.Sigmoid(*args, **kwargs)

@register_conversion(torch.ops.aten.slice.Tensor)
def Slice(*args, **kwargs):
    return tops_op.Slice(*args, **kwargs)

@register_conversion(torch.ops.aten.slice_scatter.default)
def SliceScatter(*args, **kwargs):
    return tops_op.SliceScatter(*args, **kwargs)

@register_conversion(torch.ops.aten.index.Tensor)
def Index(*args, **kwargs):
    return tops_op.Index(*args, **kwargs)

@register_conversion(torch.ops.aten.where.self)
def Where(*args, **kwargs):
    return tops_op.Where(*args, **kwargs)

@register_conversion(torch.ops.aten.select.int)
def Select(*args, **kwargs):
    return tops_op.Select(*args, **kwargs)

@register_conversion(torch.ops.aten.scatter.value)
def Scatter(*args, **kwargs):
    return tops_op.Scatter(*args, **kwargs)

@register_conversion(torch.ops.aten.zeros_like)
def zeroslike(*args, **kwargs):
    return tops_op.ZerosLike(*args, **kwargs)

@register_conversion(torch.ops.aten.ones_like)
def oneslike(*args, **kwargs):
    return tops_op.OnesLike(*args, **kwargs)

@register_conversion(torch.ops.aten.scalar_tensor.default)
def Scalar(*args, **kwargs):
    return tops_op.Scalar(*args, **kwargs)

@register_conversion(torch.ops.aten.embedding)
def Embedding(*args, **kwargs):
    return tops_op.Embedding(*args, **kwargs)

@register_conversion(torch.ops.aten.eq.Scalar)
def Eq(*args):
    return tops_op.Equal(*args)

@register_conversion(torch.ops.aten.repeat.default)
def Tile(*args, **kwargs):
    return tops_op.Tile(*args, **kwargs)

@register_conversion(torch.ops.prims.convert_element_type.default)
def ConvertElementType(*args, **kwargs):
    return tops_op.ConvertElementType(*args, **kwargs)

@register_conversion(torch.ops.aten.view_as_complex)
def ViewAsComplex(x):
    return tops_op.ViewAsComplex(x)

@register_conversion(torch.ops.aten.view_as_real)
def ViewAsReal(*args, **kwargs):
    return tops_op.ViewAsReal(*args, **kwargs)

@register_conversion(torch.ops.aten._unsafe_view.default)
def UnsafeView(a, b):
    return tops_op.UnsafeView(a, b)

@register_conversion(torch.ops.aten._log_softmax.default)
def Logsoftmax(*args, **kwargs):
    return tops_op.Logsoftmax(*args, **kwargs)

@register_conversion(torch.ops.aten.gelu.default)
def Gelu(*args, **kwargs):
    return tops_op.Gelu(*args, **kwargs)

@register_conversion(torch.ops.aten.gelu_backward.default)
def gelubackward(*args, **kwargs):
    return tops_op.GeluBackward(*args, **kwargs)

@register_conversion(torch.ops.prims.iota.default)
def Iota(*args, **kwargs):
    return tops_op.Iota(*args, **kwargs)

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
