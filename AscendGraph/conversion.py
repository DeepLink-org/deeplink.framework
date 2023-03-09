import functools
from AscendGraph.ascend_op import *
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


# @registe_conversion(torch.add)
# def add(a, b):
#     return torch.ascend.operator.add(a, b)

# @registe_conversion(torch.abs)
# def abs(a):
#     return torch.ascend.operator.abs(a)


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

@registe_pattern
class ReplacePattern1:
    def pattern(a, b):
        return torch.add(a, b)

    def replacement(a, b):
        return torch.ascend.operator.add(a, b)


# @registe_pattern
# class ReplacePattern:
#     def pattern(a):
#         v1 = torch.add(a, a)
#         return torch.add(v1, v1)

#     def replacement(a):
#         return torch.ascend.operator.mull(a, a)

# @registe_pattern
# class ReplacePattern2:
#     def pattern(a):
#         # return torch.abs(a)
#         m =  torch.ascend.operator.mull(a, a)
#         return torch.abs(m)
    
#     def replacement(a):
#         return torch.ascend.operator.mull(a, a)

    