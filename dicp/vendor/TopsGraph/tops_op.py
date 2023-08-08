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
        if torch.__version__.startswith("2.0"):
            self.shape_env = ShapeEnv() if config.use_dynamic_shapes else None
            self.fake_mode = (
                FakeTensorMode(shape_env=self.shape_env)
                if config.use_fake_tensor
                else nullcontext()
            )
        elif torch.__version__.startswith("2.1"):
            self.shape_env = ShapeEnv() if torch._dynamo.config.dynamic_shapes else None
            self.fake_mode = (
                FakeTensorMode(shape_env=self.shape_env)
                if config.fake_tensor_allow_meta
                else nullcontext()
            )
        else:
            raise ValueError(f"unsupported dicp torch version: {torch.__version__}")

    def name(self):
        return self.__name__

    def __call__(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, list):
                new_args.append([x if not hasattr(x, 'meta') else x.meta['val'] for x in arg])
            else:
                new_args.append(arg if not hasattr(arg, 'meta') else arg.meta['val'])
        new_args = tuple(new_args)
        
        fake_mode = None
        for arg in new_args:
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break
            elif isinstance(arg, list):
                for x in arg:
                    if isinstance(x, FakeTensor):
                        fake_mode = x.fake_mode
                        break
                if fake_mode is not None:
                    break
        if fake_mode is None:
            fake_mode = self.fake_mode
            
        tmp_args = []
        for arg in new_args:
            if not isinstance(arg, torch.Tensor) or isinstance(arg, FakeTensor):
                tmp_args.append(arg)
            else:
                tmp_args.append(FakeTensor.from_tensor(arg, fake_mode))
        new_args = tuple(tmp_args)

        return self.torch_op(*new_args, **kwargs)

class Add(Operator):
    def __init__(self, a, b):
        super().__init__("Add")
        self.a = a
        self.b = b
        self.torch_op = aten.add.Tensor

class AddDefalut(Operator):
    def __init__(self, a, b):
        super().__init__("Add")
        self.a = a
        self.b = b
        self.torch_op = aten.add.default

class AddScalar(Operator):
    def __init__(self, a, b):
        super().__init__("Add")
        self.a = a
        self.b = b
        self.torch_op = aten.add.Scalar

class Gemm(Operator):
    def __init__(self, a, b):
        super().__init__("Gemm")
        self.a = a
        self.b = b
        self.torch_op = aten.mm


class Abs(Operator):
    def __init__(self, a):
        super().__init__("Abs")
        self.a = a
        self.torch_op = aten.abs

class LtTensor(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Less")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.lt.Tensor


class LessEqual(Operator):
    def __init__(self, *args):
        super().__init__("LessEqual")
        self.args = args
        self.torch_op = aten.le.Scalar

class NeScalar(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("NotEqual")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.ne.Scalar

class Mul(Operator):
    def __init__(self, a, b):
        super().__init__("Mul")
        self.a = a
        self.b = b
        self.torch_op = aten.mul


class ComplexMul(Operator):
    def __init__(self, a, b):
        super().__init__("Complexmul")
        self.a = a
        self.b = b
        self.torch_op = aten.mul

class MulScalar(Operator):
    def __init__(self, a, b):
        super().__init__("Mul")
        self.a = a
        self.b = b
        self.torch_op = aten.mul.Scalar

class Div(Operator):
    def __init__(self, a, b):
        super().__init__("Div")
        self.a = a
        self.b = b
        self.torch_op = aten.div

class DivScalar(Operator):
    def __init__(self, a, b):
        super().__init__("Div")
        self.a = a
        self.b = b
        self.torch_op = aten.div.Scalar

class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("Sub")
        self.a = a
        self.b = b
        self.torch_op = aten.sub


class Sqrt(Operator):
    def __init__(self, a):
        super().__init__("Sqrt")
        self.a = a
        self.torch_op = aten.sqrt

class Square(Operator):
    def __init__(self, *args):
        super().__init__("Square")
        self.args = args
        self.torch_op = aten.square


class Exp(Operator):
    def __init__(self, a):
        super().__init__("Exp")
        self.a = a
        self.torch_op = aten.exp


class Relu(Operator):
    def __init__(self, a):
        super().__init__("Relu")
        self.a = a
        self.torch_op = aten.relu


class ReduceSum(Operator):
    def __init__(self, *args):
        super().__init__("ReduceSum")
        self.args = args
        self.torch_op = aten.sum


class ReduceMean(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("ReduceMean")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.mean


class ReduceMax(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("ReduceMax")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.amax


class Squeeze(Operator):
    def __init__(self, a, b):
        super().__init__("Squeeze")
        self.a = a
        self.b = b
        self.torch_op = aten.squeeze


class Unsqueeze(Operator):
    def __init__(self, a, b):
        super().__init__("Unsqueeze")
        self.a = a
        self.b = b
        self.torch_op = aten.unsqueeze


class Transpose(Operator):
    def __init__(self, a, b):
        super().__init__("Transpose")
        self.a = a
        self.b = b
        self.torch_op = aten.permute


class Transpose1(Operator):
    def __init__(self, a, b, c):
        super().__init__("Transpose")
        self.a = a
        self.b = b
        self.c = c
        self.torch_op = aten.transpose


class Hardswish(Operator):
    def __init__(self, a):
        super().__init__("Hardswish")
        self.a = a
        self.torch_op = aten.hardswish


class HardswishBackward(Operator):
    def __init__(self, a, b):
        super().__init__("Hardswish_Grad")
        self.a = a
        self.b = b
        self.torch_op = aten.hardswish_backward


class Clone(Operator):
    def __init__(self, *args, **kargs):
        super().__init__("Clone")
        self.args = args
        self.torch_op = aten.clone


class Copy(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Copy")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.aten.copy.default


class LiftFreshCopy(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("LiftFreshCopy")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.aten.lift_fresh_copy.default

    def __call__(self, *args, **kwargs):
        if hasattr(args[0], 'meta'):
            return args[0].meta['val']
        else:
            with FakeTensorMode():
                return torch.empty(())


class Neg(Operator):
    def __init__(self, *args):
        super().__init__("Neg")
        self.args = args
        self.torch_op = aten.neg


class Reshape(Operator):
    def __init__(self, a, b):
        super().__init__("Reshape")
        self.a = a
        self.b = b
        self.torch_op = aten.view


class Reciprocal(Operator):
    def __init__(self, a):
        super().__init__("Reciprocal")
        self.a = a
        self.torch_op = aten.reciprocal

class Rsqrt(Operator):
    def __init__(self, a):
        super().__init__("Rsqrt")
        self.a = a
        self.torch_op = aten.rsqrt

class Convolution(Operator):
    def __init__(self, *args):
        super().__init__("Convolution")
        self.args = args
        self.torch_op = aten.convolution

class ConvolutionBackward(Operator):
    def __init__(self, *args):
        super().__init__("Conv2D_Grad")
        self.args = args
        self.torch_op = aten.convolution_backward.default

class Max_pool2d_with_indices(Operator):
    def __init__(self, *args):
        super().__init__("MaxPool2D")
        self.args = args
        self.torch_op = aten.max_pool2d_with_indices


class Max_pool2d_with_indices_backward(Operator):
    def __init__(self, *args):
        super().__init__("MaxPool2D_Grad")
        self.args = args
        self.torch_op = aten.max_pool2d_with_indices_backward
        
        
class Adaptive_avg_pool2d(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("AvgPool2D")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten._adaptive_avg_pool2d.default
       
        
class Adaptive_avg_pool2d_backward(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("AvgPool2D_Grad")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten._adaptive_avg_pool2d_backward.default


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
    def __init__(self, x, idx):
        super().__init__("Getitem")
        self.x = x
        self.idx = idx
        self.torch_op = operator.getitem


class NativeDropout(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("NativeDropout")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.aten.native_dropout.default


class BatchNorm(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Batch_Norm")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten._native_batch_norm_legit_functional.default


class BatchNormBackward(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("BatchnormBackward")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.native_batch_norm_backward.default


class Softmax(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Softmax")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten._softmax.default


class Range(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Range")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.arange.start


class Dotgeneral(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Dotgeneral")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.bmm.default

class Dot(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Dot")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.dot.default

class Concatenate(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Concatenate")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.cat.default


class EmptyLike(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("EmptyLike")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.empty_like.default


class NewEmptyStrided(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("NewEmptyStrided")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.new_empty_strided.default


class Euqal(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Equal")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.eq.Tensor


class Expand(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Expand")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.expand.default


class Full(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Full")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.full.default


class FullLike(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("FullLike")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.full_like.default


class Max(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Max")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.maximum.default


class Pow(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Pow")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.pow.Tensor_Scalar


class Sigmoid(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Sigmoid")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.sigmoid.default


class Slice(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Slice")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.slice.Tensor
        
        
class SliceScatter(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("SliceScatter")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.aten.slice_scatter.default


class Index(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Index")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.index.Tensor


class Where(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Where")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.where.self


class Select(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Select")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.select.int


# scatter_value = torch.ops.aten.scatter.value(fulllike, 1, unsqueeze, -1.0);  fulllike = unsqueeze = None
class Scatter(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Scatter")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.scatter.value


class ZerosLike(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("ZerosLike")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.zeros_like


class OnesLike(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("OnesLike")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.ones_like


class Scalar(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Scalar")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.scalar_tensor.default
        

class Embedding(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Embedding")
        self.args = args
        self.kargs = kwargs
        self.torch_op = aten.embedding.default

class Equal(Operator):
    def __init__(self, *args):
        super().__init__("Equal")
        self.args = args
        self.torch_op = aten.eq.Scalar


class Tile(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Tile")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten.repeat.default


# torch.ops.prims.convert_element_type.default
class ConvertElementType(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Convert")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.prims.convert_element_type.default


class ViewAsComplex(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Complex")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.aten.view_as_complex


class ViewAsReal(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Viewasreal")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.aten.view_as_real

class UnsafeView(Operator):
    def __init__(self, a, b):
        super().__init__("Reshape")
        self.a = a
        self.b = b
        self.torch_op = aten._unsafe_view.default


class Logsoftmax(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Logsoftmax")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = aten._log_softmax.default


class Gelu(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Gelu")
        self.args = args
        self.kwarg = kwargs
        self.torch_op = aten.gelu.default


class GeluBackward(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Gelu_Grad")
        self.args = args
        self.kwarg = kwargs
        self.torch_op = aten.gelu_backward.default


class Iota(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__("Iota")
        self.args = args
        self.kwargs = kwargs
        self.torch_op = torch.ops.prims.iota.default

# TODO check if we need this wrap
@torch.fx.wrap
def ret_tuples(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b