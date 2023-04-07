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

@register_conversion(torch.ops.aten.add)
def add(a, b):
    return tops_op.Add(a, b)

@register_conversion(torch.ops.aten.abs)
def abs(a):
    return tops_op.Abs(a)

@register_conversion(torch.ops.aten.mul)
def mul(a, b):
    return tops_op.Mul(a, b)

@register_conversion(torch.ops.aten.div)
def div(a, b):
    return tops_op.Div(a, b)

@register_conversion(torch.ops.aten.sub)
def sub(a, b):
    return tops_op.Sub(a, b)

@register_conversion(torch.ops.aten.sqrt)
def sqrt(a):
    return tops_op.Sqrt(a)

@register_conversion(torch.ops.aten.square)
def square(*args):
    return tops_op.Square(*args)

@register_conversion(torch.ops.aten.reciprocal)
def reciprocal(a):
    return tops_op.Reciprocal(a)

@register_conversion(torch.ops.aten.exp)
def exp(a):
    return tops_op.Exp(a)

@register_conversion(torch.ops.aten.relu)
def relu(a):
    return tops_op.Relu(a)

@register_conversion(torch.ops.aten.sum)
def sum(*args):
    return tops_op.ReduceSum(*args)

@register_conversion(operator.getitem)
def getitem(*args, **kwargs):
    return tops_op.Getitem(*args, **kwargs)

# torch.ops.aten.squeeze.dim(,[])
# torch.ops.aten.squeeze.dims(,)
@register_conversion(torch.ops.aten.squeeze)
def squeeze(a, b):
    return tops_op.Squeeze(a, b)

@register_conversion(torch.ops.aten.unsqueeze)
def unsqueeze(a, b):
    return tops_op.Unsqueeze(a, b)

@register_conversion(torch.ops.aten.permute)
def permute(a, b):
    return tops_op.Transpose(a, b)

@register_conversion(torch.ops.aten.transpose)
def transpose(a, b, c):
    return tops_op.Transpose1(a, b, c)

@register_conversion(torch.ops.aten.clone)
def clone(*args):
    return tops_op.Copy(*args)

@register_conversion(torch.ops.aten.neg)
def neg(*args):
    return tops_op.Neg(*args)

# %mean_dim : [#users=2] = call_function[target=torch.ops.aten.mean.dim]
#                          (args = (%relu_16, [-1, -2], True), kwargs = {})
@register_conversion(torch.ops.aten.mean)
def mean(*args):
    return tops_op.ReduceMean(*args)

@register_conversion(torch.ops.aten.view)
def view(a, b):
    return tops_op.Reshape(a, b)

@register_conversion(torch.ops.aten.convolution)
def convolution(*args):
    return tops_op.Convolution(*args)

@register_conversion(torch.ops.aten.convolution_backward)
def convolution_backward(*args):
    return tops_op.ConvolutionBackward(*args)

@register_conversion(torch.ops.aten.le.Scalar)
def le(*args):
    return tops_op.LessEqual(*args)

@register_conversion(torch.ops.aten.max_pool2d_with_indices)
def max_pool2d_with_indices(*args):
    return tops_op.Max_pool2d_with_indices(*args)

@register_conversion(torch.ops.aten.max_pool2d_with_indices_backward)
def max_pool2d_with_indices(*args):
    return tops_op.Max_pool2d_with_indices_backward(*args)

@register_conversion(torch.ops.aten.gather)
def gather(*args):
    return tops_op.Gather(*args)

@register_conversion(torch.ops.aten.log)
def log(*args):
    return tops_op.Log(*args)

@register_conversion(torch.ops.aten.amax)
def max(*args, **kwargs):
    return tops_op.ReduceMax(*args, **kwargs)

@register_conversion(torch.ops.aten.mm)
def gemm(*args, **kwargs):
    return tops_op.Gemm(*args, **kwargs)

@register_conversion(torch.ops.aten._native_batch_norm_legit_functional.default)
def bathnorm(*args, **kwargs):
    return tops_op.BathNorm(*args, **kwargs)

@register_conversion(torch.ops.aten.native_batch_norm_backward.default)
def bathnorm(*args, **kwargs):
    return tops_op.BathNormBackward(*args, **kwargs)

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
class ReplacePattern1:
    def pattern(a):
        return torch.ops.aten.rsqrt.default(a)

    def replacement(a):
        return torch.ops.aten.reciprocal.default(torch.ops.aten.sqrt.default(a))

@register_pattern
class ReplacePattern2:
    def pattern(a, b, c):
        return torch.ops.aten.addmm.default(a, b, c)

    def replacement(a, b, c):
        return torch.ops.aten.add.default(a, torch.ops.aten.mm.default(b, c))

# %var: [#users=2] = call_function[target=torch.ops.aten.var.correction]
#                                      (args = (%convolution_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})

@register_pattern
class ReplacePattern3:
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
class ReplacePattern4:
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
class ReplacePattern5:
    def pattern(a):
        return torch.ops.aten.t.default(a)

    def replacement(inputs):
        return torch.ops.aten.transpose(inputs, 0, 1)

# convolution_backward =
#torch.ops.aten.convolution_backward.default(
# getitem_102,
# relu_15,
# primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
# [True, True, False])
#;  getitem_102 = primals_58 = None
# getitem_105: f32[4, 512, 7, 7] = convolution_backward[0]
# getitem_106: f32[512, 512, 3, 3] = convolution_backward[1];  convolution_backward = None
@register_pattern
class ReplacePattern5:
    def pattern(
        grad_output,
        inputs,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        return torch.ops.aten.convolution_backward.default(
            grad_output,
            inputs,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )

    def replacement(
        grad_output,
        inputs,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        # grad_output.dims()
        # TODO
        # grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
        grad_bias = torch.ops.aten.sum(grad_output, [2])
        grad_inp, grad_weight, _ = torch.ops.aten.convolution_backward(
            grad_output,
            inputs,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            [output_mask[0], output_mask[1], False],
        )
        return tops_op.ret_tri_tuples(grad_inp, grad_weight, grad_bias)
