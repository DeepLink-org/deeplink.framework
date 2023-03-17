import torch
import functools
from . import tops_op
from abc import ABC, abstractmethod
from torch.fx import Proxy


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

def registe_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_conversion,
        aten_fn,
    )

@registe_conversion(torch.ops.aten.add)
def add(a, b):
    return tops_op.Add(a, b)

@registe_conversion(torch.ops.aten.abs)
def abs(a):
    return tops_op.Abs(a)

@registe_conversion(torch.ops.aten.mul)
def mul(a, b):
    return tops_op.Mul(a, b)

@registe_conversion(torch.ops.aten.div)
def div(a, b):
    return tops_op.Div(a, b)

@registe_conversion(torch.ops.aten.sub)
def sub(a, b):
    return tops_op.Sub(a, b)

@registe_conversion(torch.ops.aten.sqrt)
def sqrt(a):
    return tops_op.Sqrt(a)

@registe_conversion(torch.ops.aten.exp)
def exp(a):
    return tops_op.Exp(a)

@registe_conversion(torch.ops.aten.relu)
def relu(a):
    return tops_op.Relu(a)

@registe_conversion(torch.ops.aten.sum)
def sum(*args):
    return tops_op.ReduceSum(*args)

#torch.ops.aten.squeeze.dim(,[])
#torch.ops.aten.squeeze.dims(,)
@registe_conversion(torch.ops.aten.squeeze)
def squeeze(a,b):
    return tops_op.Squeeze(a,b)

@registe_conversion(torch.ops.aten.unsqueeze)
def unsqueeze(a,b):
    return tops_op.Unsqueeze(a,b)

@registe_conversion(torch.ops.aten.permute)
def permute(a, b):
    return tops_op.Transpose(a,b)

@registe_conversion(torch.ops.aten.clone)
def clone(*args):
    return tops_op.Copy(*args)

# %mean_dim : [#users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_16, [-1, -2], True), kwargs = {})
@registe_conversion(torch.ops.aten.mean)
def mean(*args):
    return tops_op.Mean(*args)

@registe_conversion(torch.ops.aten.view)
def view(a, b):
    return tops_op.Reshape(a,b)

@registe_conversion(torch.ops.aten.convolution)
def convolution(*args):
    return tops_op.Convolution(*args)

@registe_conversion(torch.ops.aten.max_pool2d_with_indices)
def convolution(*args):
    return tops_op.Max_pool2d_with_indices(*args)

# pattern
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

# %rsqrt_default_1 : [#users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
@registe_pattern
class ReplacePattern1:
    def pattern(a, b):
        return torch.ops.aten.rsqrt.default(a, b)
    def replacement(a, b):
        return tops_op.reciprocal(tops_op.sqrt(a, b))
        #return tops_op.Reciprocal(r1)
        #return tops_op.add(a, b)
@registe_pattern
class ReplacePattern2:
    def pattern(a):
        return torch.ops.aten.rsqrt.default(a)
    def replacement(a):
        return tops_op.reciprocal(tops_op.sqrt(a))

@registe_pattern
class ReplacePattern3:
    def pattern(a, b, c):
        return torch.ops.aten.addmm.default(a, b, c)
    def replacement(a, b, c):
        #return  tops_op.addmm(a,b,c)
        return tops_op.add(a, tops_op.gemm(b, c))
