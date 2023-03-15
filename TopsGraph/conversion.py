import torch
import functools
from torch.tops import *
from abc import ABC, abstractmethod
from torch.fx import Proxy
import operator

conversions = {}
patterns = []
aten = torch.ops.aten
prims = torch.ops.prims

def _registe_conversion(
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

def registe_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _registe_conversion,
        aten_fn,
    )

@registe_conversion(torch.ops.aten.add)
def add(a, b):
    return torch.tops.operator.Add(a, b)

@registe_conversion(torch.ops.aten.abs)
def abs(a):
    return torch.tops.operator.Abs(a)

@registe_conversion(torch.ops.aten.mul)
def mul(a, b):
    return torch.tops.operator.Mul(a, b)

@registe_conversion(torch.ops.aten.div)
def div(a, b):
    return torch.tops.operator.Div(a, b)

@registe_conversion(torch.ops.aten.sub)
def sub(a, b):
    return torch.tops.operator.Sub(a, b)

@registe_conversion(torch.ops.aten.sqrt)
def sqrt(a):
    return torch.tops.operator.Sqrt(a)

@registe_conversion(torch.ops.aten.exp)
def exp(a):
    return torch.tops.operator.Exp(a)

@registe_conversion(torch.ops.aten.relu)
def relu(a):
    return torch.tops.operator.Relu(a)

@registe_conversion(torch.ops.aten.sum)
def sum(*args):
    return torch.tops.operator.ReduceSum(*args)

@registe_conversion(operator.getitem)
def getitem(*args, **kwargs):
    return torch.tops.operator.Getitem(*args, **kwargs)

# TODO
#torch.ops.aten.squeeze.dim(,[])
#torch.ops.aten.squeeze.dims(,)
@registe_conversion(torch.ops.aten.squeeze)
def squeeze(a,b):
    return torch.tops.operator.Squeeze(a,b)

@registe_conversion(torch.ops.aten.unsqueeze)
def unsqueeze(a,b):
    return torch.tops.operator.Unsqueeze(a,b)

@registe_conversion(torch.ops.aten.permute)
def permute(a, b):
    return torch.tops.operator.Transpose(a,b)

@registe_conversion(torch.ops.aten.clone)
def clone(*args):
    return torch.tops.operator.Copy(*args)

@registe_conversion(torch.ops.aten.neg)
def neg(*args):
    return torch.tops.operator.Neg(*args)

# %mean_dim : [#users=2] = call_function[target=torch.ops.aten.mean.dim]
#                          (args = (%relu_16, [-1, -2], True), kwargs = {})
@registe_conversion(torch.ops.aten.mean)
def mean(*args, **kwargs):
    return torch.tops.operator.ReduceMean(*args, **kwargs)

#%var_mean_correction_4 : [#users=2] = call_function[target=torch.ops.aten.var_mean.correction]
#                                      (args = (%convolution_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
@registe_conversion(torch.ops.aten.var_mean.correction)
def varmean(*args, **kwargs):
    return torch.tops.operator.ReduceMean(*args, **kwargs)

@registe_conversion(torch.ops.aten.view)
def view(a, b):
    return torch.tops.operator.Reshape(a,b)

@registe_conversion(torch.ops.aten.convolution)
def convolution(*args):
    return torch.tops.operator.Convolution(*args)

@registe_conversion(torch.ops.aten.le.Scalar)
def convolution(*args):
    return torch.tops.operator.LessEqual(*args)

#TODO max_pool2d with getitem
@registe_conversion(torch.ops.aten.max_pool2d_with_indices)
def convolution(*args):
    return torch.tops.operator.Max_pool2d_with_indices(*args)

@registe_conversion(torch.ops.aten.gather)
def gather(*args):
    return torch.tops.operator.Gather(*args)

@registe_conversion(torch.ops.aten.log)
def log(*args):
    return torch.tops.operator.Log(*args)

#%amax_default : [#users=1] = call_function[target=torch.ops.aten.amax.default]
#                             (args = (%primals_1, [1], True), kwargs = {})
@registe_conversion(torch.ops.aten.amax)
def max(*args, **kwargs):
    return torch.tops.operator.ReduceMax(*args, **kwargs)

# Patterns
def registe_pattern(Pattern):
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

# %rsqrt_default_1 : [#users=2] = call_function[target=torch.ops.aten.rsqrt.default]
#                                 (args = (%add_6,), kwargs = {})
@registe_pattern
class ReplacePattern1:
    def pattern(a, b):
        return torch.ops.aten.rsqrt.default(a, b)
    def replacement(a, b):
        return torch.tops.operator.reciprocal(torch.tops.operator.sqrt(a, b))

@registe_pattern
class ReplacePattern2:
    def pattern(a):
        return torch.ops.aten.rsqrt.default(a)
    def replacement(a):
        return torch.tops.operator.reciprocal(torch.tops.operator.sqrt(a))

@registe_pattern
class ReplacePattern3:
    def pattern(a, b, c):
        return torch.ops.aten.addmm.default(a, b, c)
    def replacement(a, b, c):
        return torch.tops.operator.add(a, torch.tops.operator.gemm(b, c))