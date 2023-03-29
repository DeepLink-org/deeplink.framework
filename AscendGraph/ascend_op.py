import torch

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

    def __call__(self, *args, **kwds):
        self.torch_op = aten.add


class AddV2(Operator):
    def __init__(self, a, b):
        super().__init__("addv2")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.add


class MatMul(Operator):
    def __init__(self, a, b):
        super().__init__("matmul")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.matmul


class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("sub")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.sub


class Mul(Operator):
    def __init__(self, a, b):
        super().__init__("mul")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.mul


class Div(Operator):
    def __init__(self, a, b):
        super().__init__("div")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.div


class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.abs


class Rsqrt(Operator):
    def __init__(self, a):
        super().__init__("rsqrt")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.rsqrt


class Log(Operator):
    def __init__(self, a):
        super().__init__("log")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.log


class Exp(Operator):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.exp


class Neg(Operator):
    def __init__(self, a):
        super().__init__("neg")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.neg


class Relu(Operator):
    def __init__(self, a):
        super().__init__("relu")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.relu


class Sum(Operator):
    def __init__(self, a):
        super().__init__("sum")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.sum


class ReduceSumD(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("reducesum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        self.torch_op = aten.sum


class Copy(Operator):
    def __init__(self, a):
        super().__init__("copy")
        self.a = a

    def __call__(self, *args, **kwds):
        self.torch_op = aten.clone


class Unsqueeze(Operator):
    def __init__(self, x, dims):
        super().__init__("unsqueeze")
        self.x = x
        self.dims = dims

    def __call__(self, *args, **kwds):
        self.torch_op = aten.unsqueeze


class Squeeze(Operator):
    def __init__(self, x, dims):
        super().__init__("squeeze")
        self.x = x
        self.dims = dims

    def __call__(self, *args, **kwds):
        self.torch_op = aten.squeeze


class Permute(Operator):
    def __init__(self, x, dims):
        super().__init__("permute")
        self.x = x
        self.dims = dims

    def __call__(self, *args, **kwds):
        self.torch_op = aten.permute


class ReduceMean(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("reducemean")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        self.torch_op = aten.mean


class Amax(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("amax")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        self.torch_op = aten.amax


class GatherD(Operator):
    def __init__(self, x, dims, index):
        super().__init__("gatherd")
        self.x = x
        self.dims = dims
        self.index = index

    def __call__(self, *args, **kwds):
        self.torch_op = aten.gather


class Where(Operator):
    def __init__(self, condition, a, b):
        super().__init__("where")
        self.condition = condition
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.where


class Ne(Operator):
    def __init__(self, x, scalar):
        super().__init__("ne")
        self.x = x
        self.scalar = scalar

    def __call__(self, *args, **kwds):
        self.torch_op = aten.ne


class LessEqual(Operator):
    def __init__(self, a, b):
        super().__init__("lessequal")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        self.torch_op = aten.le


class Conv2D(Operator):
    def __init__(self, input, weight, bias, stride, padding,
                 dilation, transposed, output_padding, groups):
        super().__init__("convolution")
        self.input = input
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

    def __call__(self, *args, **kwds):
        self.torch_op = aten.convolution


class TranShape(Operator):
    def __init__(self, x, shape):
        super().__init__("transhape")
        self.x = x
        self.shape = shape

    def __call__(self, *args, **kwds):
        self.torch_op = aten.reshape


class Identity(Operator):
    def __init__(self, x, idx):
        super().__init__("identity")
        self.x = x
        self.idx = idx

    def __call__(self, *args, **kwds):
        self.torch_op = aten.tensor


class Pad(Operator):
    def __init__(self, x, padding):
        super().__init__("pad")
        self.x = x
        self.padding = padding

    def __call__(self, *args, **kwds):
        self.torch_op = aten.tensor


class MaxPoolWithArgmax(Operator):
    def __init__(self, input, kernel_size, stride):
        super().__init__("maxpoolwithargmax")
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, *args, **kwds):
        self.torch_op = aten.max_pool2d


class BroadcastTo(Operator):
    def __init__(self, input, shape):
        super().__init__("broadcastto")
        self.input = input
        self.shape = shape

    def __call__(self, *args, **kwds):
        self.torch_op = aten.broadcast_to


class SquareSumV1(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("squaresum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        self.torch_op = aten.tensor


class Shape(Operator):
    def __init__(self, x):
        super().__init__("shape")
        self.x = x

    def __call__(self, *args, **kwds):
        self.torch_op = aten.tensor


@torch.fx.wrap
def addv2(a, b) -> torch.Tensor:
    return aten.add(a, b)

@torch.fx.wrap
def matmul(a, b) -> torch.Tensor:
    return aten.matmul(a, b)

@torch.fx.wrap
def pad(x, padding) -> torch.Tensor:
    return aten.tensor(x)

@torch.fx.wrap
def maxpoolwithargmax(input, kernel_size, stride) -> torch.Tensor:
    return aten.max_pool2d(input, kernel_size, stride)

@torch.fx.wrap
def broadcastto(x, shape) -> torch.Tensor:
    return aten.broadcast_to(x, shape)

@torch.fx.wrap
def squaresum(x, dims, keepdim) -> torch.Tensor:
    return aten.tensor(x)

@torch.fx.wrap
def shape(x) -> torch.Tensor:
    return aten.tensor(x)

