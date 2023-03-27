import functools
import operator
import torch
import third_party.DICP.AscendGraph.ascend_op as ascend_op
from abc import ABC, abstractmethod

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


def registe_pattern(Pattern):
    patterns.append(Pattern)
    return Pattern


class BaseReplacePattern(ABC):
    @abstractmethod
    def pattern(*args, **kwargs):
        pass
    @abstractmethod
    def replacement(*args, **kwargs):
        pass

@registe_conversion(torch.ops.aten.add)
def add(a, b):
    return ascend_op.Add(a, b)

@registe_conversion(torch.ops.aten.sub)
def sub(a, b):
    return ascend_op.Sub(a, b)

@registe_conversion(torch.ops.aten.mul)
def mul(a, b):
    return ascend_op.Mul(a, b)

@registe_conversion(torch.ops.aten.div)
def div(a, b):
    return ascend_op.Div(a, b)

@registe_conversion(torch.ops.aten.convolution)
def convolution(input, weight, bias, stride, padding,
                dilation, transposed, output_padding, groups):
    return ascend_op.Conv2D(input, weight, bias, stride, padding,
                dilation, transposed, output_padding, groups)

@registe_conversion(torch.ops.aten.abs)
def abs(a):
    return ascend_op.Abs(a)

@registe_conversion(torch.ops.aten.rsqrt)
def rsqrt(a):
    return ascend_op.Rsqrt(a)

@registe_conversion(torch.ops.aten.log)
def log(a):
    return ascend_op.Log(a)

@registe_conversion(torch.ops.aten.exp)
def exp(a):
    return ascend_op.Exp(a)

@registe_conversion(torch.ops.aten.neg)
def neg(a):
    return ascend_op.Neg(a)

@registe_conversion(torch.ops.aten.relu)
def relu(a):
    return ascend_op.Relu(a)

@registe_conversion(torch.ops.aten.sum.default)
def sum(a):
    return ascend_op.Sum(a)

@registe_conversion(torch.ops.aten.sum.dim_IntList)
def sumdim(x, dims, keepdim):
    return ascend_op.ReduceSumD(x, dims, keepdim)

@registe_conversion(torch.ops.aten.clone)
def clone(a):
    return ascend_op.Copy(a)

@registe_conversion(torch.ops.aten.ne)
def ne(x, scalar):
    return ascend_op.Ne(x, scalar)

@registe_conversion(torch.ops.aten.le)
def le(a, b):
    return ascend_op.LessEqual(a, b)

@registe_conversion(torch.ops.aten.unsqueeze)
def unsqueeze(x, dims):
    return ascend_op.Unsqueeze(x, dims)

@registe_conversion(torch.ops.aten.squeeze)
def squeeze(x, dims):
    return ascend_op.Squeeze(x, dims)

@registe_conversion(torch.ops.aten.permute)
def permute(x, dims):
    return ascend_op.Permute(x, dims)

@registe_conversion(torch.ops.aten.mean)
def mean(x, dims, keepdim):
    return ascend_op.ReduceMean(x, dims, keepdim)

@registe_conversion(torch.ops.aten.amax)
def amax(x, dims, keepdim):
    return ascend_op.Amax(x, dims, keepdim)

@registe_conversion(torch.ops.aten.gather)
def gather(x, dims, index):
    return ascend_op.GatherD(x, dims, index)

@registe_conversion(torch.ops.aten.where)
def where(condition, a, b):
    return ascend_op.Where(condition, a, b)

@registe_conversion(torch.ops.aten.view)
def view(x, shape):
    return ascend_op.TranShape(x, shape)

@registe_conversion(operator.getitem)
def identity(x, idx):
    return ascend_op.Identity(x, idx)

@registe_pattern
class ReplaceAddmm:
    def pattern(input, mat1, mat2):
        return torch.ops.aten.addmm.default(input, mat1, mat2)

    def replacement(input, mat1, mat2):
        mul = ascend_op.matmul(mat1, mat2)
        return ascend_op.addv2(input, mul)

@registe_pattern
class ReplaceMaxPool:
    def pattern(input, kernel_size, stride, padding=0):
        return torch.ops.aten.max_pool2d_with_indices.default(input, kernel_size, stride, padding)

    def replacement(input, kernel_size, stride, padding=0):
        pad = ascend_op.pad(input, padding)
        return ascend_op.maxpoolwithargmax(pad, kernel_size, stride)

@registe_pattern
class ReplaceVarMean:
    def pattern(input, dims):
        return torch.ops.aten.var_mean.correction(input, dims, correction=0, keepdim=True)

    def replacement(input, dims):
        mean = torch.ops.aten.mean(input, dims, True)
        shape = ascend_op.shape(input)
        broadcast = ascend_op.broadcastto(mean, shape)
        sub = torch.ops.aten.sub(input, broadcast)
        square = ascend_op.squaresum(sub, dims, True)
        return torch.ops.aten.mul(square, 1 / (64 - 1))

