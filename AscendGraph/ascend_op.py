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

    def __call__(self, *args, **kwds):
        return torch.add(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'], args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class AddV2(Operator):
    def __init__(self, a, b):
        super().__init__("addv2")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        return torch.add(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'], args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class MatMul(Operator):
    def __init__(self, a, b):
        super().__init__("matmul")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        return torch.matmul(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'], args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("sub")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        return torch.sub(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'], args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Mul(Operator):
    def __init__(self, a, b):
        super().__init__("mul")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        return torch.mul(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'], args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Div(Operator):
    def __init__(self, a, b):
        super().__init__("div")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        return torch.div(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'], args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.abs(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Rsqrt(Operator):
    def __init__(self, a):
        super().__init__("rsqrt")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.rsqrt(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Log(Operator):
    def __init__(self, a):
        super().__init__("log")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.log(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Exp(Operator):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.exp(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Neg(Operator):
    def __init__(self, a):
        super().__init__("neg")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.neg(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Relu(Operator):
    def __init__(self, a):
        super().__init__("relu")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.relu(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Sum(Operator):
    def __init__(self, a):
        super().__init__("sum")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.sum(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class ReduceSumD(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("reducesum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        torch.sum(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                  args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                  args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'])


class Copy(Operator):
    def __init__(self, a):
        super().__init__("copy")
        self.a = a

    def __call__(self, *args, **kwds):
        torch.clone(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Unsqueeze(Operator):
    def __init__(self, x, dims):
        super().__init__("unsqueeze")
        self.x = x
        self.dims = dims

    def __call__(self, *args, **kwds):
        torch.unsqueeze(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                        args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Squeeze(Operator):
    def __init__(self, x, dims):
        super().__init__("squeeze")
        self.x = x
        self.dims = dims

    def __call__(self, *args, **kwds):
        torch.squeeze(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                      args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Permute(Operator):
    def __init__(self, x, dims):
        super().__init__("permute")
        self.x = x
        self.dims = dims

    def __call__(self, *args, **kwds):
        torch.permute(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                      args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class ReduceMean(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("reducemean")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        torch.mean(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                   args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                   args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'])


class Amax(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("amax")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        torch.amax(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                   args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                   args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'])


class GatherD(Operator):
    def __init__(self, x, dims, index):
        super().__init__("gatherd")
        self.x = x
        self.dims = dims
        self.index = index

    def __call__(self, *args, **kwds):
        torch.gather(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                     args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                     args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'])


class Where(Operator):
    def __init__(self, condition, a, b):
        super().__init__("where")
        self.condition = condition
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        torch.where(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                    args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                    args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'])


class Ne(Operator):
    def __init__(self, x, scalar):
        super().__init__("ne")
        self.x = x
        self.scalar = scalar

    def __call__(self, *args, **kwds):
        torch.ne(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                 args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class LessEqual(Operator):
    def __init__(self, a, b):
        super().__init__("lessequal")
        self.a = a
        self.b = b

    def __call__(self, *args, **kwds):
        torch.le(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                 args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


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
        torch.convolution(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                          args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                          args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'],
                          args[3] if not hasattr(args[3], 'meta') else args[3].meta['val'],
                          args[4] if not hasattr(args[4], 'meta') else args[4].meta['val'],
                          args[5] if not hasattr(args[5], 'meta') else args[5].meta['val'],
                          args[6] if not hasattr(args[6], 'meta') else args[6].meta['val'],
                          args[7] if not hasattr(args[7], 'meta') else args[7].meta['val'],
                          args[8] if not hasattr(args[8], 'meta') else args[8].meta['val'])


class TranShape(Operator):
    def __init__(self, x, shape):
        super().__init__("transhape")
        self.x = x
        self.shape = shape

    def __call__(self, *args, **kwds):
        torch.reshape(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                      args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class Identity(Operator):
    def __init__(self, x, idx):
        super().__init__("identity")
        self.x = x
        self.idx = idx

    def __call__(self, *args, **kwds):
        torch.tensor(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Pad(Operator):
    def __init__(self, x, padding):
        super().__init__("pad")
        self.x = x
        self.padding = padding

    def __call__(self, *args, **kwds):
        torch.tensor(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                     args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class MaxPoolWithArgmax(Operator):
    def __init__(self, input, kernel_size, stride):
        super().__init__("maxpoolwithargmax")
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, *args, **kwds):
        torch.max_pool2d(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                         args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'],
                         args[2] if not hasattr(args[2], 'meta') else args[2].meta['val'])


class BroadcastTo(Operator):
    def __init__(self, input, shape):
        super().__init__("broadcastto")
        self.input = input
        self.shape = shape

    def __call__(self, *args, **kwds):
        torch.broadcast_to(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'],
                           args[1] if not hasattr(args[1], 'meta') else args[1].meta['val'])


class SquareSumV1(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("squaresum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, *args, **kwds):
        torch.tensor(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


class Shape(Operator):
    def __init__(self, x):
        super().__init__("shape")
        self.x = x

    def __call__(self, *args, **kwds):
        torch.tensor(args[0] if not hasattr(args[0], 'meta') else args[0].meta['val'])


@torch.fx.wrap
def addv2(a, b) -> torch.Tensor:
    return torch.add(a, b)

@torch.fx.wrap
def matmul(a, b) -> torch.Tensor:
    return torch.matmul(a, b)

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

