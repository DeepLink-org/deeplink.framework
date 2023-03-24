import torch
import torch.fx
from typing import Tuple, Dict, Optional, Iterable, Any, Iterator, Callable

class Operator():
    __name__: str
    def __init__(self, name_):
        super().__init__()
        self.__name__ = name_

    def __call__(self, *args, **kwds):
        return self.__name__

class Add(Operator):
    def __init__(self, a, b):
        super().__init__("add")
        self.a = a
        self.b = b

class Gemm(Operator):
    def __init__(self, a, b):
        super().__init__("gemm")
        self.a = a
        self.b = b

class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a

class LessEqual(Operator):
    def __init__(self, *args):
        super().__init__("LessEqual")
        self.args = args

class Mul(Operator):
    def __init__(self, a, b):
        super().__init__("mul")
        self.a = a
        self.b = b

class Div(Operator):
    def __init__(self, a, b):
        super().__init__("div")
        self.a = a
        self.b = b

class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("sub")
        self.a = a
        self.b = b

class Sqrt(Operator):
    def __init__(self, a):
        super().__init__("sqrt")
        self.a = a

class Square(Operator):
    def __init__(self, *args):
        super().__init__("square")
        self.args = args

class Exp(Operator):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a

class Relu(Operator):
    def __init__(self, a):
        super().__init__("relu")
        self.a = a

class ReduceSum(Operator):
    def __init__(self, *args):
        super().__init__("reduceSum")
        self.args = args

class ReduceMean(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("reducemean")
        self.args = args
        self.args = kwargs

class ReduceMax(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("reducemax")
        self.args = args
        self.args = kwargs

class Squeeze(Operator):
    def __init__(self, a, b):
        super().__init__("squeeze")
        self.a = a
        self.b = b

class Unsqueeze(Operator):
    def __init__(self, a, b):
        super().__init__("unsqueeze")
        self.a = a
        self.b = b

class Transpose(Operator):
    def __init__(self, a, b):
        super().__init__("transpose")
        self.a = a
        self.b = b

class Copy(Operator):
    def __init__(self, *args):
        super().__init__("copy")
        self.args = args

class Neg(Operator):
    def __init__(self, *args):
        super().__init__("neg")
        self.args = args

class Reshape(Operator):
    def __init__(self, a, b):
        super().__init__("reshape")
        self.a = a
        self.b = b

class Reciprocal(Operator):
    def __init__(self, a):
        super().__init__("reciprocal")
        self.a = a

class Convolution(Operator):
    def __init__(self, *args):
        super().__init__("convolution")
        self.args = args

class Max_pool2d_with_indices(Operator):
    def __init__(self, *args):
        super().__init__("max_pool2d_with_indices")
        self.args = args

class Gather(Operator):
    def __init__(self, *args):
        super().__init__("Gather")
        self.args = args

class Log(Operator):
    def __init__(self, *args):
        super().__init__("Log")
        self.args = args

class Getitem(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("getitem")
        self.args = args
        self.args = kwargs

# For pattern replacement
@torch.fx.wrap
def sqrt(a, b) -> torch.Tensor:
    return torch.sqrt(a, b)
@torch.fx.wrap
def sqrt1(a) -> torch.Tensor:
    return torch.sqrt(a)

@torch.fx.wrap
def reciprocal(a) -> torch.Tensor:
    return torch.reciprocal(a)

@torch.fx.wrap
def add(a, b) -> torch.Tensor:
    return torch.add(a,b)

@torch.fx.wrap
def gemm(a, b) -> torch.Tensor:
    return torch.gemm(a,b)

@torch.fx.wrap
def tuple(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b