import torch
from typing import Tuple
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
        self.shape_env = ShapeEnv() if torch._dynamo.config.dynamic_shapes else None
        self.fake_mode = (
            FakeTensorMode(shape_env=self.shape_env)
            if config.fake_tensor_allow_meta
            else nullcontext()
        )
    
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
        try:
            for arg in new_args:
                if not isinstance(arg, torch.Tensor) or isinstance(arg, FakeTensor):
                    tmp_args.append(arg)
                else:
                    tmp_args.append(FakeTensor.from_tensor(arg, fake_mode))
        except Exception  as e:
            print(e)
            import pdb;pdb.set_trace()
        new_args = tuple(tmp_args)
        #new_args = tuple(arg if not isinstance(arg, torch.Tensor) else FakeTensor.from_tensor(arg, fake_mode) for arg in new_args)
        return self.torch_op(*new_args, **kwargs)


class Add(Operator):
    def __init__(self, a, b):
        super().__init__("add")
        self.a = a
        self.b = b
        self.torch_op = aten.add


class AddV2(Operator):
    def __init__(self, a, b):
        super().__init__("addv2")
        self.a = a
        self.b = b
        self.torch_op = aten.add


class MatMul(Operator):
    def __init__(self, a, b):
        super().__init__("mm")
        self.a = a
        self.b = b
        self.torch_op = aten.matmul


class BatchMatMul(Operator):
    def __init__(self, a, b):
        super().__init__("bmm")
        self.a = a
        self.b = b
        self.torch_op = aten.bmm


class Sub(Operator):
    def __init__(self, a, b):
        super().__init__("sub")
        self.a = a
        self.b = b
        self.torch_op = aten.sub


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


class Abs(Operator):
    def __init__(self, a):
        super().__init__("abs")
        self.a = a
        self.torch_op = aten.abs


class Rsqrt(Operator):
    def __init__(self, a):
        super().__init__("rsqrt")
        self.a = a
        self.torch_op = aten.rsqrt


class Log(Operator):
    def __init__(self, a):
        super().__init__("log")
        self.a = a
        self.torch_op = aten.log


class Exp(Operator):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a
        self.torch_op = aten.exp


class Neg(Operator):
    def __init__(self, a):
        super().__init__("neg")
        self.a = a
        self.torch_op = aten.neg


class Relu(Operator):
    def __init__(self, a):
        super().__init__("relu")
        self.a = a
        self.torch_op = aten.relu


class Silu(Operator):
    def __init__(self, a):
        super().__init__("silu")
        self.a = a
        self.torch_op = aten.silu


class Transpose(Operator):
    def __init__(self, input, dim0, dim1):
        super().__init__("transpose")
        self.input = input
        self.dim0 = dim0
        self.dim1 = dim1
        self.torch_op = aten.transpose


class ToCopy(Operator):
    def __init__(self, x, dtype, layout, device):
        super().__init__("_to_copy")
        self.x = x
        self.dtype = dtype
        self.layout = layout
        self.device = device
        self.torch_op = aten._to_copy


class Softmax(Operator):
    def __init__(self, x, dim, half_to_float):
        super().__init__("_softmax")
        self.x = x
        self.dim = dim
        self.half_to_float = half_to_float
        self.torch_op = aten._softmax


class Sum(Operator):
    def __init__(self, a):
        super().__init__("sum")
        self.a = a
        self.torch_op = aten.sum


class ReduceSumD(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("sum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim
        self.torch_op = aten.sum


class Copy(Operator):
    def __init__(self, a, memory_format):
        super().__init__("clone")
        self.a = a
        self.memory_format = memory_format
        self.torch_op = aten.clone


class CopyInner(Operator):
    def __init__(self, dst, src):
        super().__init__("copy")
        self.dst = dst
        self.src = src
        self.torch_op = aten.copy


class Unsqueeze(Operator):
    def __init__(self, x, dims):
        super().__init__("unsqueeze")
        self.x = x
        self.dims = dims
        self.torch_op = aten.unsqueeze


class Squeeze(Operator):
    def __init__(self, x, dims):
        super().__init__("squeeze")
        self.x = x
        self.dims = dims
        self.torch_op = aten.squeeze


class Permute(Operator):
    def __init__(self, x, dims):
        super().__init__("permute")
        self.x = x
        self.dims = dims
        self.torch_op = aten.permute


class ExpandD(Operator):
    def __init__(self, x, dims):
        super().__init__("expand")
        self.x = x
        self.dims = dims

    def __call__(self, x, dims):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        return x.expand(dims)


class ScatterElement(Operator):
    def __init__(self, x, dims, index, value):
        super().__init__("scatter")
        self.x = x
        self.dims = dims
        self.index = index
        self.value = value
        self.torch_op = aten.scatter


class ReduceMean(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("mean")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim
        self.torch_op = aten.mean


class Var(Operator):
    def __init__(self, x, dims, correction, keepdim):
        super().__init__("var")
        self.x = x
        self.dims = dims
        self.correction = correction
        self.keepdim = keepdim

    def __call__(self, x, dims, correction, keepdim):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        prod_value = 1
        for i, s in enumerate(x.size()):
            if i not in dims:
                prod_value = prod_value * s
        assert(prod_value - correction != 0)
        div_value = 1.0 / (prod_value - correction)
        meanVal = aten.mean.dim(x, dims, keepdim)
        broadCast = aten.broadcast_to(meanVal, x.shape)
        subVal = aten.sub(x, broadCast)
        square = aten.square(subVal)
        sumSquare = aten.sum(square, dims, keepdim)
        return aten.mul(sumSquare, div_value)


class Amax(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("amax")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim
        self.torch_op = aten.amax


class GatherD(Operator):
    def __init__(self, x, dims, index):
        super().__init__("gather")
        self.x = x
        self.dims = dims
        self.index = index
        self.torch_op = aten.gather


class Where(Operator):
    def __init__(self, condition, a, b):
        super().__init__("where")
        self.condition = condition
        self.a = a
        self.b = b
        self.torch_op = aten.where


class Convert(Operator):
    def __init__(self, x, dtype):
        super().__init__("convert_element_type")
        self.x = x
        self.dtype = dtype
        self.torch_op = torch.ops.prims.convert_element_type


class Embedding(Operator):
    def __init__(self, weight, indices):
        super().__init__("embedding")
        self.weight = weight
        self.indices = indices
        self.torch_op = aten.embedding


class Sigmoid(Operator):
    def __init__(self, x):
        super().__init__("sigmoid")
        self.x = x
        self.torch_op = aten.sigmoid


class Pow(Operator):
    def __init__(self, x, exp):
        super().__init__("pow")
        self.x = x
        self.exp = exp
        self.torch_op = aten.pow


class Ne(Operator):
    def __init__(self, x, scalar):
        super().__init__("ne")
        self.x = x
        self.scalar = scalar
        self.torch_op = aten.ne


class LessEqual(Operator):
    def __init__(self, a, b):
        super().__init__("le")
        self.a = a
        self.b = b
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
        self.torch_op = aten.convolution


class TranShape(Operator):
    def __init__(self, x, shape):
        super().__init__("view")
        self.x = x
        self.shape = shape
        self.torch_op = aten.reshape


class Identity(Operator):
    def __init__(self, x, idx):
        super().__init__("getitem")
        self.x = x
        self.idx = idx

    def __call__(self, x, idx):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        x = x[idx]
        if hasattr(x, 'meta'):
            x = x.meta['val']
        return aten.clone(x)


class Pad(Operator):
    def __init__(self, x, padding):
        super().__init__("pad")
        self.x = x
        self.padding = padding

    def __call__(self, x, padding):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        shape = x.size()
        for i in range(len(shape)):
            shape[i] += padding
        return aten.zeros(shape)


class MaxPoolWithArgmax(Operator):
    def __init__(self, input, kernel_size, stride):
        super().__init__("maxpoolwithargmax")
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride
        self.torch_op = aten.max_pool2d


class SquareSumV1(Operator):
    def __init__(self, x, dims, keepdim):
        super().__init__("squaresum")
        self.x = x
        self.dims = dims
        self.keepdim = keepdim

    def __call__(self, x, dims, keepdim):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        square = aten.square(x)
        return aten.sum(square, dims, keepdim)


class FullLike(Operator):
    def __init__(self, x, value):
        super().__init__("zeros_like")
        #TODO! only handles cases for zero
        assert value == 0
        self.x = x
        self.value = value
        self.torch_op = aten.full_like


class Full(Operator):
    def __init__(self, dims, value):
        super().__init__("full")
        self.dims = dims
        self.value = value
        self.torch_op = aten.full


class AddMm(Operator):
    def __init__(self, input, mat1, mat2):
        super().__init__("addmm")
        self.input = input
        self.mat1 = mat1
        self.mat2 = mat2
        self.torch_op = aten.addmm


class MaxPool(Operator):
    def __init__(self, input, kernel_size, stride, padding):
        super().__init__("max_pool2d_with_indices")
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.torch_op = aten.max_pool2d_with_indices


class MaxPoolGradWithArgmaxV1(Operator):
    def __init__(self, input, grad, argmax, ksize, strides, pads, dilation, ceil_mode):
        super().__init__("max_pool2d_with_indices_backward")
        self.input = input
        self.grad = grad
        self.argmax = argmax
        self.ksize = ksize
        self.strides = strides
        self.pads = pads
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.torch_op = aten.max_pool2d_with_indices_backward


class ConvBackward(Operator):
    def __init__(self, grad, input, weight, bias,
                stride, padding, dilation, transposed,
                output_padding, groups, output_masks):
        super().__init__("convolution_backward")
        self.input = input
        self.grad = grad
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.output_masks = output_masks
        self.torch_op = aten.convolution_backward


class T(Operator):
    def __init__(self, input):
        super().__init__("t")
        self.input = input
        self.torch_op = aten.t


class LogSoftmax(Operator):
    def __init__(self, x, dim, half_to_float):
        super().__init__("log_softmax")
        self.x = x
        self.dim = dim
        self.half_to_float = half_to_float
        self.torch_op = aten._log_softmax


class LogSoftmaxBackward(Operator):
    def __init__(self, grad_output, output, dim, input_dtype):
        super().__init__("log_softmax_backward")
        self.grad_output = grad_output
        self.output = output
        self.dim = dim
        self.input_dtype = input_dtype
        self.torch_op = aten._log_softmax_backward_data


class NLLLossForward(Operator):
    def __init__(self, x, target, weight, reduction, ignore_index):
        super().__init__("nll_loss_forward")
        self.x = x
        self.target = target
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.torch_op = aten.nll_loss_forward


class NLLLossBackward(Operator):
    def __init__(self, grad_output, x, target, weight, reduction, ignore_index,
                 total_weight):
        super().__init__("nll_loss_backward")
        self.grad_output = grad_output
        self.x = x
        self.target = target
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.total_weight = total_weight
        self.torch_op = aten.nll_loss_backward


class BatchNorm(Operator):
    def __init__(self, x, weight, bias, running_mean, running_var,
                 train, momentum, eps):
        super().__init__("native_batch_norm_legit_functional")
        self.x = x
        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var
        self.train = train
        self.momentum = momentum
        self.eps = eps
        self.torch_op = aten._native_batch_norm_legit_functional


class BatchNormBackward(Operator):
    def __init__(self, grad_out, x, weight, running_mean, running_var,
                 save_mean, save_invstd, train, eps, grad_input_mask):
        super().__init__("native_batch_norm_backward")
        self.grad_out = grad_out
        self.x = x
        self.weight = weight
        self.running_mean = running_mean
        self.running_var = running_var
        self.save_mean = save_mean
        self.save_invstd = save_invstd
        self.train = train
        self.eps = eps
        self.grad_input_mask = grad_input_mask
        self.torch_op = aten.native_batch_norm_backward


class ThresholdBackward(Operator):
    def __init__(self, grad_output, x, threshold):
        super().__init__("threshold_backward")
        self.grad_output = grad_output
        self.x = x
        self.threshold = threshold
        self.torch_op = aten.threshold_backward


class ZerosLike(Operator):
    def __init__(self, x):
        super().__init__("zeros_like")
        self.x = x
        self.torch_op = aten.zeros_like


class ViewAsComplex(Operator):
    def __init__(self, x):
        super().__init__("view_as_complex")
        self.x = x
        self.torch_op = aten.view_as_complex
        

class ViewAsReal(Operator):
    def __init__(self, x):
        super().__init__("view_as_real")
        self.x = x
        self.torch_op = aten.view_as_real
        

class Slice(Operator):
    def __init__(self, x, dim, start, end, step):
        super().__init__("slice")
        self.x = x
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step
        self.torch_op = aten.slice
        

class Cat(Operator):
    def __init__(self, x, dim=0):
        super().__init__("cat")
        self.x = x
        self.dim = dim
        self.torch_op = aten.cat
        

class Select(Operator):
    def __init__(self, x, dim, index):
        super().__init__("select")
        self.x = x
        self.dim = dim
        self.index = index
        self.torch_op = aten.select


@torch.fx.wrap
def addv2(a, b) -> torch.Tensor:
    if hasattr(a, 'meta'):
        a = a.meta['val']
    if hasattr(b, 'meta'):
        b = b.meta['val']
    return aten.add(a, b)

@torch.fx.wrap
def pad(x, padding) -> torch.Tensor:
    if hasattr(x, 'meta'):
        x = x.meta['val']
    shape = list(x.size())
    for i in range(len(shape)):
        shape[i] = shape[i] + padding
    return aten.zeros(shape)

@torch.fx.wrap
def maxpoolwithargmax(input, kernel_size, stride) -> torch.Tensor:
    return aten.max_pool2d(input, kernel_size, stride)

@torch.fx.wrap
def squaresum(x, dims, keepdim) -> torch.Tensor:
    square = aten.square(x)
    return aten.sum(square, dims, keepdim)

@torch.fx.wrap
def conv2dbackpropfilter(grad, input, weight, bias,
        stride, padding, dilation, transposed, output_padding, groups, output_masks) -> torch.Tensor:
    output_masks = [True, False, False]
    if hasattr(grad, 'meta'):
        grad = grad.meta['val']
    if hasattr(input, 'meta'):
        input = input.meta['val']
    if hasattr(weight, 'meta'):
        weight = weight.meta['val']
    return aten.convolution_backward(grad, input, weight, bias,
            stride, padding, dilation, transposed,
            output_padding, groups, output_masks)

@torch.fx.wrap
def conv2dbackpropinput(grad, input, weight, bias,
        stride, padding, dilation, transposed, output_padding, groups, output_masks) -> torch.Tensor:
    output_masks = [False, True, False]
    if hasattr(grad, 'meta'):
        grad = grad.meta['val']
    if hasattr(input, 'meta'):
        input = input.meta['val']
    if hasattr(weight, 'meta'):
        weight = weight.meta['val']
    return aten.convolution_backward(grad, input, weight, bias,
            stride, padding, dilation, transposed,
            output_padding, groups, output_masks)

@torch.fx.wrap
def biasaddgrad(grad, input, weight, bias,
        stride, padding, dilation, transposed, output_padding, groups, output_masks) -> torch.Tensor:
    output_masks = [False, False, True]
    if hasattr(grad, 'meta'):
        grad = grad.meta['val']
    if hasattr(input, 'meta'):
        input = input.meta['val']
    if hasattr(weight, 'meta'):
        weight = weight.meta['val']
    return aten.convolution_backward(grad, input, weight, bias,
            stride, padding, dilation, transposed,
            output_padding, groups, output_masks)

@torch.fx.wrap
def ret_triple(a, b, c) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return a, b, c

@torch.fx.wrap
def ret_tuple(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b

