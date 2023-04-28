import torch
import torch.fx
from typing import Tuple
import operator
 
from contextlib import nullcontext
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._functorch import config
aten = torch.ops.aten

class Operator():
    __name__: str

    def __init__(self, name_):
        super().__init__()
        self.__name__ = name_
        self.shape_env = ShapeEnv() if config.use_dynamic_shapes else None
        self.fake_mode = (
            FakeTensorMode(shape_env=self.shape_env)
            if config.use_fake_tensor
            else nullcontext()
        )

    def name(self):
        return self.__name__

    def __call__(self, *args, **kwargs):
        new_args = tuple(arg if not hasattr(arg, 'meta') else arg.meta['val'] for arg in args)
        fake_mode = None
        for arg in new_args:
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break
        if fake_mode is None:
            fake_mode = self.fake_mode
        new_args = tuple(arg if not isinstance(arg, torch.Tensor) else FakeTensor.from_tensor(arg, fake_mode) for arg in new_args)
        return self.torch_op(*new_args, **kwargs)

class Add(Operator):
    def __init__(self, a, b):
        super().__init__("add")
        self.a = a
        self.b = b
        self.torch_op = aten.add.Tensor

'''
class Addmm(Operator):
    def __init__(self, a, b, c):
        super().__init__("addmm")
        self.a = a
        self.b = b
        self.c = c
        self.torch_op = aten.addmm
'''

class AddDefalut(Operator):
    def __init__(self, a, b):
        super().__init__("add")
        self.a = a
        self.b = b
        self.torch_op = aten.add.default

class Gemm(Operator):
    def __init__(self, a, b):
        super().__init__("Gemm")
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
        super().__init__("lessequal")
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
        super().__init__("ReduceSum")
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
        super().__init__("ReduceMax")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.amax


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
        super().__init__("clone")
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
        super().__init__("Reciprocal")
        self.a = a
        self.torch_op = aten.reciprocal


class Convolution(Operator):
    def __init__(self, *args):
        super().__init__("Convolution")
        self.args = args
        self.torch_op = aten.convolution

class ConvolutionBackward(Operator):
    def __init__(self, *args):
        super().__init__("Conv2D_Grad")
        self.args = args
        self.torch_op = aten.convolution_backward

class Max_pool2d_with_indices(Operator):
    def __init__(self, *args):
        super().__init__("max_pool2d_with_indices")
        self.args = args
        self.torch_op = aten.max_pool2d_with_indices


class Max_pool2d_with_indices_backward(Operator):
    def __init__(self, *args):
        super().__init__("max_pool2d_with_indices_backward")
        self.args = args
        self.torch_op = aten.max_pool2d_with_indices_backward


class Gather(Operator):
    def __init__(self, *args):
        super().__init__("Gather")
        self.args = args
        self.torch_op = aten.gather


class Log(Operator):
    def __init__(self, *args):
        super().__init__("log")
        self.args = args
        self.torch_op = aten.log


class Getitem(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Getitem")
        self.args = args
        self.args = kwargs
        self.torch_op = operator.getitem


class BatchNorm(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Batch_Norm")
        self.args = args
        self.args = kwargs
        self.torch_op = aten._native_batch_norm_legit_functional.default


class BatchNormBackward(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("batchnorm_backward")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.native_batch_norm_backward.default


class Softmax(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("softmax")
        self.args = args
        self.args = kwargs
        self.torch_op = aten._softmax.default


class Range(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("range")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.arange.start


class Dot(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("dot")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.bmm.default


class Concatenate(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("concatenate")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.cat.default


class EmptyLike(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("empty_like")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.empty_like.default


class Euqal(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("eauql")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.eq.Tensor


class Expand(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Expand")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.expand.default


class Full(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("fulllike")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.full.default


class FullLike(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("fulllike")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.full_like.default


class Max(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("max")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.maximum.default


class Pow(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("pow")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.pow.Tensor_Scalar


class Sigmoid(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("sigmoid")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.sigmoid.default


class Slice(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("slice")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.slice.Tensor


class Select(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Select")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.where.self


class AddGrad(Operator):
    def __init__(self, a, b):
        super().__init__("addgrad")
        self.a = a
        self.b = b
        self.torch_op = aten.sum


# scatter_value = torch.ops.aten.scatter.value(fulllike, 1, unsqueeze, -1.0);  fulllike = unsqueeze = None
class Scatter(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("scatter")
        self.args = args
        self.args = kwargs
        self.torch_op = aten.scatter.value


class Zeros(Operator):
    def __init__(self, *args):
        super().__init__("Zeros")
        self.args = args
        self.torch_op = aten.zeros


# TODO check if we need this wrap
@torch.fx.wrap
def ret_tuples(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b


@torch.fx.wrap
def ret_tri_tuples(a, b, c) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return a, b, c


@torch.fx.wrap
def warpaddgrad(a, b) -> torch.Tensor:
    return AddGrad(a, b)
