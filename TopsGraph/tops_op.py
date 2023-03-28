import torch
import torch.fx
from typing import Tuple
import operator

aten = torch.ops.aten

class Operator():
    __name__: str

    def __init__(self, name_):
        super().__init__()
        self.__name__ = name_

    def __call__(self, *args, **kwargs):
        new_args = tuple(arg if not hasattr(arg, 'meta') else arg.meta['val'] for arg in args)
        return self.torch_op(*new_args, **kwargs)

class Add(Operator):

    def __init__(self, a, b):
        super().__init__("add")
        self.a = a
        self.b = b
        self.torch_op = aten.add


class Gemm(Operator):
    def __init__(self, a, b):
        super().__init__("gemm")
        self.a = a
        self.b = b
        self.torch_op = aten.mm


class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a
        self.torch_op = aten.abs


class LessEqual(Operator):
    def __init__(self, *args):
        super().__init__("LessEqual")
        self.args = args
        self.torch_op = aten.le.Scalar


class Mul(Operator):
    def __init__(self, a, b):
        super().__init__("mul")
        self.a = a
        self.b = b
        self.torch_op = aten.mul


class Div(Operator):
    def __init__(self, a, b):
        super().__init__("div")
        self.a = a
        self.b = b
        self.torch_op = aten.div


class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("sub")
        self.a = a
        self.b = b
        self.torch_op = aten.sub


class Sqrt(Operator):
    def __init__(self, a):
        super().__init__("sqrt")
        self.a = a
        self.torch_op = aten.sqrt

class Square(Operator):
    def __init__(self, *args):
        super().__init__("square")
        self.args = args
        self.torch_op = aten.square


class Exp(Operator):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a
        self.torch_op = aten.exp


class Relu(Operator):
    def __init__(self, a):
        super().__init__("relu")
        self.a = a
        self.torch_op = aten.relu


class ReduceSum(Operator):
    def __init__(self, *args):
        super().__init__("reduceSum")
        self.args = args
        self.torch_op = aten.sum


class ReduceMean(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("reducemean")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.mean


class ReduceMax(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("reducemax")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.max


class Squeeze(Operator):
    def __init__(self, a, b):
        super().__init__("squeeze")
        self.a = a
        self.b = b
        self.torch_op = aten.squeeze


class Unsqueeze(Operator):
    def __init__(self, a, b):
        super().__init__("unsqueeze")
        self.a = a
        self.b = b
        self.torch_op = aten.unsqueeze


class Transpose(Operator):
    def __init__(self, a, b):
        super().__init__("transpose")
        self.a = a
        self.b = b
        self.torch_op = aten.permute


class Transpose1(Operator):
    def __init__(self, a, b, c):
        super().__init__("transpose")
        self.a = a
        self.b = b
        self.c = c
        self.torch_op = aten.transpose


class Copy(Operator):
    def __init__(self, *args):
        super().__init__("copy")
        self.args = args
        self.torch_op = aten.clone


class Neg(Operator):
    def __init__(self, *args):
        super().__init__("neg")
        self.args = args
        self.torch_op = aten.neg


class Reshape(Operator):
    def __init__(self, a, b):
        super().__init__("reshape")
        self.a = a
        self.b = b
        self.torch_op = aten.view


class Reciprocal(Operator):
    def __init__(self, a):
        super().__init__("reciprocal")
        self.a = a
        self.torch_op = aten.reciprocal


class Convolution(Operator):
    def __init__(self, *args):
        super().__init__("convolution")
        self.args = args
        self.torch_op = aten.convolution


class Max_pool2d_with_indices(Operator):
    def __init__(self, *args):
        super().__init__("max_pool2d_with_indices")
        self.args = args
        self.torch_op = aten.max_pool2d_with_indices


class Gather(Operator):
    def __init__(self, *args):
        super().__init__("Gather")
        self.args = args
        self.torch_op = aten.gather


class Log(Operator):
    def __init__(self, *args):
        super().__init__("Log")
        self.args = args
        self.torch_op = aten.log


class Getitem(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("getitem")
        self.args = args
        self.args = kwargs
        self.torch_op = operator.getitem


class BathNorm(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("bathnorm")
        self.args = args
        self.args = kwargs
        self.torch_op = aten._native_batch_norm_legit_functional.default


# TODO check if we need this wrap
@torch.fx.wrap
def ret_tuples(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b
