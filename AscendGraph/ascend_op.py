import torch

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

class AddV2(Operator):
    def __init__(self, a, b):
        super().__init__("addv2")
        self.a = a
        self.b = b

class MatMul(Operator):
    def __init__(self, a, b):
        super().__init__("matmul")
        self.a = a
        self.b = b

class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("sub")
        self.a = a
        self.b = b

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

class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a

class Rsqrt(Operator):
    def __init__(self, a):
        super().__init__("rsqrt")
        self.a = a

class Log(Operator):
    def __init__(self, a):
        super().__init__("log")
        self.a = a

class Exp(Operator):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a

class Neg(Operator):
    def __init__(self, a):
        super().__init__("neg")
        self.a = a

class Relu(Operator):
    def __init__(self, a):
        super().__init__("relu")
        self.a = a

class Sum(Operator):
    def __init__(self, a):
        super().__init__("sum")
        self.a = a

class ReduceSumD(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("reducesum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

class Copy(Operator):
    def __init__(self, a):
        super().__init__("copy")
        self.a = a

class Unsqueeze(Operator):
    def __init__(self, x, dims):
        super().__init__("unsqueeze")
        self.x = x
        self.dims = dims

class Squeeze(Operator):
    def __init__(self, x, dims):
        super().__init__("squeeze")
        self.x = x
        self.dims = dims

class Permute(Operator):
    def __init__(self, x, dims):
        super().__init__("permute")
        self.x = x
        self.dims = dims

class ReduceMean(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("reducemean")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

class Amax(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("amax")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

class GatherD(Operator):
    def __init__(self, x, dims, index):
        super().__init__("gatherd")
        self.x = x
        self.dims = dims
        self.index = index

class Where(Operator):
    def __init__(self, condition, a, b):
        super().__init__("where")
        self.condition = condition
        self.a = a
        self.b = b

class Ne(Operator):
    def __init__(self, x, scalar):
        super().__init__("ne")
        self.x = x
        self.scalar = scalar

class LessEqual(Operator):
    def __init__(self, a, b):
        super().__init__("lessequal")
        self.a = a
        self.b = b

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

class TranShape(Operator):
    def __init__(self, x, shape):
        super().__init__("transhape")
        self.x = x
        self.shape = shape

class Identity(Operator):
    def __init__(self, x, idx):
        super().__init__("identity")
        self.x = x
        self.idx = idx

class Pad(Operator):
    def __init__(self, x, padding):
        super().__init__("pad")
        self.x = x
        self.padding = padding

class MaxPoolWithArgmax(Operator):
    def __init__(self, input, kernel_size, stride):
        super().__init__("maxpoolwithargmax")
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride

class BroadcastTo(Operator):
    def __init__(self, input, shape):
        super().__init__("broadcastto")
        self.input = input
        self.shape = shape

class SquareSumV1(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("squaresum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

class Shape(Operator):
    def __init__(self, x):
        super().__init__("shape")
        self.x = x

@torch.fx.wrap
def mul(a, b) -> torch.Tensor:
    return torch.mul(a, b)

@torch.fx.wrap
def div(a, b) -> torch.Tensor:
    return torch.div(a, b)

@torch.fx.wrap
def abs(a) -> torch.Tensor:
    return torch.abs(a)

@torch.fx.wrap
def rsqrt(a) -> torch.Tensor:
    return torch.rsqrt(a)

@torch.fx.wrap
def log(a) -> torch.Tensor:
    return torch.log(a)

@torch.fx.wrap
def exp(a) -> torch.Tensor:
    return torch.exp(a)

@torch.fx.wrap
def neg(a) -> torch.Tensor:
    return torch.neg(a)

@torch.fx.wrap
def lessequal(a, b) -> torch.Tensor:
    return torch.le(a, b)

@torch.fx.wrap
def relu(a) -> torch.Tensor:
    return torch.relu(a)

@torch.fx.wrap
def sum(a) -> torch.Tensor:
    return torch.sum(a)

@torch.fx.wrap
def copy(a) -> torch.Tensor:
    return torch.clone(a)

@torch.fx.wrap
def unsqueeze(x, dims) -> torch.Tensor:
    return torch.unsqueeze(x, dims)

@torch.fx.wrap
def squeeze(x, dims) -> torch.Tensor:
    return torch.squeeze(x, dims)

@torch.fx.wrap
def permute(x, dims) -> torch.Tensor:
    return torch.permute(x, dims)

@torch.fx.wrap
def reducemean(x, dims, keepdim) -> torch.Tensor:
    return torch.mean(x, dims, keepdim)

@torch.fx.wrap
def reducesum(x, dims, keepdim) -> torch.Tensor:
    return torch.sum(x, dims, keepdim)

@torch.fx.wrap
def amax(x, dims, keepdim) -> torch.Tensor:
    return torch.amax(x, dims, keepdim)

@torch.fx.wrap
def gatherd(x, dims, index) -> torch.Tensor:
    return torch.gather(x, dims, index)

@torch.fx.wrap
def where(condition, a, b) -> torch.Tensor:
    return torch.where(condition, a, b)

@torch.fx.wrap
def ne(x, scalar) -> torch.Tensor:
    return torch.ne(x, scalar)

@torch.fx.wrap
def add(a, b) -> torch.Tensor:
    return torch.add(a, b)

@torch.fx.wrap
def addv2(a, b) -> torch.Tensor:
    return torch.add(a, b)

@torch.fx.wrap
def matmul(a, b) -> torch.Tensor:
    return torch.matmul(a, b)

@torch.fx.wrap
def sub(a, b) -> torch.Tensor:
    return torch.sub(a, b)

@torch.fx.wrap
def convolution(input, weight, bias, stride, padding,
                dilation, transposed, output_padding, groups) -> torch.Tensor:
    return torch.convolution(input, weight, bias, stride, padding,
                dilation, transposed, output_padding, groups)

@torch.fx.wrap
def transhape(x, shape) -> torch.Tensor:
    return torch.reshape(x, shape)

@torch.fx.wrap
def identity(x, idx) -> torch.Tensor:
    return torch.tensor(x)

@torch.fx.wrap
def pad(x, padding) -> torch.Tensor:
    return torch.tensor(x)

@torch.fx.wrap
def maxpoolwithargmax(input, kernel_size, stride) -> torch.Tensor:
    return torch.max_pool2d(input, kernel_size, stride)

@torch.fx.wrap
def broadcastto(x, shape) -> torch.Tensor:
    return torch.broadcast_to(x, shape)

@torch.fx.wrap
def squaresum(x, dims, keepdim) -> torch.Tensor:
    return torch.tensor(x)

@torch.fx.wrap
def shape(x) -> torch.Tensor:
    return torch.tensor(x)

