import functools
import operator
import _operator
import torch
import dicp.vendor.AscendGraph.ascend_op as ascend_op
from abc import ABC, abstractmethod
from collections import defaultdict

conversions = defaultdict(list)
patterns = []
backend_patterns = []
aten = torch.ops.aten
prims = torch.ops.prims

def args_kwargs_unchange(args, kwargs):
    return args, kwargs

def _registe_conversion(
    aten_fn, decomp_fn, process_args_kwargs_fn=None
):
    register_op_singleton_flag = isinstance(decomp_fn, type) and issubclass(decomp_fn, ascend_op.Operator)
    if register_op_singleton_flag:
        wrapped = (decomp_fn.get_singleton(),
                   args_kwargs_unchange if process_args_kwargs_fn is None else process_args_kwargs_fn)
    else:
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
                # if other_fn not in conversions:
                aten_fn.append(other_fn)

    for fn in aten_fn:
        conversions[fn].append(wrapped)
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

# Backend Patterns
def register_backend_pattern(pattern):
    backend_patterns.append(pattern)
    return pattern


class BaseReplacePattern(ABC):
    @abstractmethod
    def pattern(*args, **kwargs):
        pass
    @abstractmethod
    def replacement(*args, **kwargs):
        pass

@registe_conversion(torch.ops.aten.arange.default)
def arange(end, start=0, step=1, device='cpu', pin_memory=False):
    return ascend_op.Arange(start, end, step)

@registe_conversion(torch.ops.aten.eq)
def eq(a, b):
    return ascend_op.Eq(a, b)

@registe_conversion(torch.ops.aten.add)
def add(a, b):
    return ascend_op.Add(a, b)

@registe_conversion(torch.ops.aten.sub)
def sub(a, b):
    return ascend_op.Sub(a, b)

@registe_conversion(torch.ops.aten.rsub)
def rsub(a, b):
    return ascend_op.Rsub(a, b)

@registe_conversion(torch.ops.aten.mul)
def mul(a, b):
    return ascend_op.Mul(a, b)

@registe_conversion(torch.ops.aten.div)
def div(a, b):
    return ascend_op.Div(a, b)

@registe_conversion(torch.ops.aten.maximum)
def maximum(a, b):
    return ascend_op.Max(a, b)

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

@registe_conversion(torch.ops.aten.sqrt)
def sqrt(a):
    return ascend_op.Sqrt(a)

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

@registe_conversion(torch.ops.aten.silu)
def silu(a):
    return ascend_op.Silu(a)

@registe_conversion(torch.ops.aten._softmax)
def _softmax(x, dim, half_to_float):
    return ascend_op.Softmax(x, dim, half_to_float)

@registe_conversion(torch.ops.aten._to_copy.default)
def _to_copy(x, dtype=None, layout=torch.strided, device='cpu'):
    return ascend_op.ToCopy(x, dtype, layout, device)

@registe_conversion(torch.ops.aten.sum.default)
def sum(a):
    return ascend_op.Sum(a)

@registe_conversion(torch.ops.aten.sum.dim_IntList)
def sumdim(x, dims, keepdim = True):
    return ascend_op.ReduceSumD(x, dims, keepdim)

@registe_conversion(torch.ops.aten.clone)
def clone(a, memory_format = torch.contiguous_format):
    return ascend_op.Copy(a, memory_format)

@registe_conversion(torch.ops.aten.copy_)
def copy_(dst, src):
    return ascend_op.Copy_(dst, src)

@registe_conversion(torch.ops.aten.copy)
def copy(dst, src):
    return ascend_op.CopyInner(dst, src)

@registe_conversion(torch.ops.prims.convert_element_type)
def convert_element_type(x, dtype):
    return ascend_op.Convert(x, dtype)

@registe_conversion(torch.ops.aten.embedding)
def embedding(weight, indices, padding_idx=-1):
    return ascend_op.Embedding(weight, indices, padding_idx)

@registe_conversion(torch.ops.aten.sigmoid)
def sigmoid(x):
    return ascend_op.Sigmoid(x)

@registe_conversion(torch.ops.aten.pow)
def pow(x, exp):
    return ascend_op.Pow(x, exp)

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

@registe_conversion(torch.ops.aten.cumsum)
def cumsum(x, dim, dtype=None):
    return ascend_op.CumSum(x, dim, dtype)

@registe_conversion(torch.ops.aten.mm)
def matmul(a, b):
    return ascend_op.MatMul(a, b)

@registe_conversion(torch.ops.aten.sort)
def sort(x, dim=-1, descending=False):
    return ascend_op.Sort(x, dim, descending)

@registe_conversion(torch.ops.aten.topk)
def topk(x, k, dim=-1, largest=True, sorted=True):
    return ascend_op.TopK(x, k, dim, largest, sorted)

@registe_conversion(torch.ops.aten.scatter)
def scatter(x, dims, index, value):
    return ascend_op.ScatterElement(x, dims, index, value)

@registe_conversion(torch.ops.aten.mean)
def mean(x, dims=[], keepdim=False):
    return ascend_op.ReduceMean(x, dims, keepdim)

@registe_conversion(torch.ops.aten.var)
def var(x, dims, correction, keepdim):
    return ascend_op.Var(x, dims, correction, keepdim)

@registe_conversion(torch.ops.aten.amax)
def amax(x, dims, keepdim):
    return ascend_op.Amax(x, dims, keepdim)

@registe_conversion(torch.ops.aten.gather)
def gather(x, dims, index):
    return ascend_op.GatherD(x, dims, index)

@registe_conversion(torch.ops.aten.where)
def where(condition, a, b):
    return ascend_op.Where(condition, a, b)

@registe_conversion(torch.ops.aten._unsafe_view)
def unsafe_view(x, shape):
    return ascend_op.TranShape(x, shape)

@registe_conversion(_operator.mul)
def inmul(a, b):
    return ascend_op.InMul(a, b)

@registe_conversion(_operator.ge)
def inge(a, b):
    return ascend_op.InGe(a, b)

@registe_conversion(_operator.add)
def inadd(a, b):
    return ascend_op.InAdd(a, b)

@registe_conversion(torch.ops.aten.sym_size)
def symsize(x, dim):
    return ascend_op.SymSize(x, dim)

@registe_conversion(operator.getitem)
def identity(x, idx):
    return ascend_op.Identity(x, idx)

@registe_conversion(torch.ops.aten.full_like)
def fulllike(x, value, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.FullLike(x, value, dtype, layout, device, pin_memory, memory_format)

@registe_conversion(torch.ops.aten.lift_fresh_copy)
def lift_fresh_copy(tensor_constant):
    return ascend_op.LiftFreshCopy(tensor_constant)

@registe_conversion(torch.ops.aten.lt)
def lt(a, b):
    return ascend_op.Lt(a, b)

@registe_conversion(torch.ops.aten.masked_fill)
def masked_fill(x, mask, value):
    return ascend_op.MaskedFill(x, mask, value)

@registe_conversion(torch.ops.aten.empty)
def empty(size, dtype=torch.int64, layout=torch.strided, device='cpu'):
    return ascend_op.Empty(size, dtype, layout, device)

@registe_conversion(torch.ops.aten.index)
def index(x, index):
    return ascend_op.IndexSelect(x, 0, index, None)

@registe_conversion(torch.ops.aten.index_select)
def index_arg2_(x, index):
    return ascend_op.IndexSelect(x, 0, index, 2)

@registe_conversion(torch.ops.aten.index_select)
def index_arg3_(x, dim, index):
    return ascend_op.IndexSelect(x, dim, index, 3)

@registe_conversion(torch.ops.aten.fill)
def fill(x, value):
    return ascend_op.Fill(x, value)

@registe_conversion(torch.ops.aten.ones)
def ones(shape, dtype=torch.int64, device='cpu', pin_memory=False):
    return ascend_op.Ones(shape)

@registe_conversion(torch.ops.aten.new_ones)
def new_ones(x, shape, dtype=torch.int64, layout=None, device='cpu', pin_memory=False):
    return ascend_op.NewOnes(x, shape)

@registe_conversion(torch.ops.aten.repeat_interleave)
def repeat_interleave(repeats, output_size = 1):
    return ascend_op.RepeatInterleave(repeats, output_size)

@registe_conversion(torch.ops.aten.full.default)
def full(dims, value, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.Full(dims, value, dtype, layout, device, pin_memory, memory_format)


@registe_conversion(torch.ops.aten.max_pool2d_with_indices)
def maxpool2d(input, kernel_size, stride, padding):
    return ascend_op.MaxPool(input, kernel_size, stride, padding)

@registe_conversion(torch.ops.aten.max_pool2d_with_indices_backward)
def maxpool2dbackward(grad, input, kernel_size, stride, padding, dilation, ceil_mode, index):
    return ascend_op.MaxPoolGradWithArgmaxV1(input, grad, index, kernel_size, stride, padding, dilation, ceil_mode)

@registe_conversion(torch.torch.ops.aten.addmm)
def addmm(input, mat1, mat2):
    return ascend_op.AddMm(input, mat1, mat2)

@registe_conversion(torch.ops.aten.convolution_backward)
def convolutionbackward(grad, input, weight, bias,
                stride, padding, dilation, transposed,
                output_padding, groups, output_masks):
    return ascend_op.ConvBackward(grad, input, weight, bias,
                stride, padding, dilation, transposed,
                output_padding, groups, output_masks)

@registe_conversion(torch.ops.aten._log_softmax.default)
def log_softmax(x, dim, half_to_float):
    return ascend_op.LogSoftmax(x, dim, half_to_float)

@registe_conversion(torch.ops.aten._log_softmax_backward_data.default)
def log_softmax_backward_data(grad_output, output, dim, input_dtype):
    return ascend_op.LogSoftmaxBackward(grad_output, output, dim, input_dtype)

@registe_conversion(torch.ops.aten.nll_loss_forward.default)
def nll_loss_forward(x, target, weight, reduction, ignore_index):
    return ascend_op.NLLLossForward(x, target, weight, reduction, ignore_index)

@registe_conversion(torch.ops.aten.nll_loss_backward.default)
def nll_loss_backward(grad_output, x, target, weight, reduction, ignore_index, total_weight):
    return ascend_op.NLLLossBackward(grad_output, x, target, weight, reduction,
                                     ignore_index, total_weight)

@registe_conversion(torch.ops.aten._native_batch_norm_legit_functional.default)
def _native_batch_norm_legit_functional(x, weight, bias, running_mean, running_var,
                                        train, momentum, eps):
    return ascend_op.BatchNorm(x, weight, bias, running_mean, running_var, train,
                               momentum, eps)

@registe_conversion(torch.ops.aten.threshold_backward.default)
def threshold_backward(grad_output, x, threshold):
    return ascend_op.ThresholdBackward(grad_output, x, threshold)

@registe_conversion(torch.ops.aten.native_batch_norm_backward.default)
def native_batch_norm_backward(grad_out, x, weight, running_mean, running_var,
        save_mean, save_invstd, train, eps, grad_input_mask):
    return ascend_op.BatchNormBackward(grad_out, x, weight, running_mean, running_var,
            save_mean, save_invstd, train, eps, grad_input_mask)

@registe_conversion(torch.ops.aten.zeros_like.default)
def zeros_like(x, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.ZerosLike(x)

@registe_conversion(torch.ops.aten.view_as_complex.default)
def view_as_complex(x):
    return ascend_op.ViewAsComplex(x)

@registe_conversion(torch.ops.aten.view_as_real.default)
def view_as_real(x):
    return ascend_op.ViewAsReal(x)

@registe_conversion(torch.ops.aten.slice.Tensor)
def slice(x, dim=0, start=None, end=None, step=1):
    return ascend_op.Slice(x, dim, start, end, step)

@registe_conversion(torch.ops.aten.stack)
def stack(x, dim):
    return ascend_op.Stack(x, dim)

@registe_conversion(torch.ops.aten.cat.default)
def cat(x, dim=0):
    return ascend_op.Cat(x, dim)

@registe_conversion(torch.ops.aten.select.int)
def select(x, dim, index):
    return ascend_op.Select(x, dim, index)

@registe_conversion(torch.ops.aten.lt.Tensor)
def lt(x, y):
    return ascend_op.Lt(x, y)
  
@registe_conversion(torch.ops.aten.masked_fill.Scalar)
def masked_fill(x, y, value):
    return ascend_op.MaskedFill(x, y, value)
  
@registe_conversion(torch.ops.aten.rsub.Scalar)
def rsub(x, value):
    return ascend_op.Rsub(x, value)

@registe_conversion(torch.ops.aten.index.Tensor)
def index(*args, **kwargs):
    return ascend_op.Index(*args, **kwargs)

@registe_conversion(torch.ops.aten._unsafe_view.default)
def unsafe_view(a, b):
    return ascend_op.UnsafeView(a, b)
 
@registe_conversion(torch.ops.aten.slice_backward.default)
def slice_backward(grad, input_shape, dim, start, end, step):
    return ascend_op.SliceBackward(grad, input_shape, dim, start, end, step)

@registe_conversion(torch.ops.aten.empty_like.default)
def empty_like(x, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.EmptyLike(x, dtype, layout, device, pin_memory, memory_format)
  
@registe_conversion(torch.ops.aten.fill.Scalar)
def fill_scalar(x, value):
    return ascend_op.FillScalar(x, value)

@registe_conversion(torch.ops.aten._softmax_backward_data.default)
def softmax_backward_data(grad_output, output, dim, input_dtype):
    return ascend_op.SoftmaxBackward(grad_output, output, dim, input_dtype)

@registe_conversion(torch.ops.aten.lift_fresh_copy.default)
def LiftFreshCopy(*args, **kwargs):
    return ascend_op.LiftFreshCopy(*args, **kwargs)

@registe_conversion(torch.ops.aten.maximum.default)
def Maximum(x, y):
    return ascend_op.Maximum(x, y)

@registe_conversion(torch.ops.aten.eq.Tensor)
def Eq(x, y):
    return ascend_op.Eq(x, y)

@registe_conversion(torch.ops.aten.t.default)
def t(input):
    return ascend_op.T(input)

@registe_conversion(torch.ops.aten.mm)
def matmul(a, b):
    return ascend_op.MatMul(a, b)

@registe_conversion(torch.ops.aten.transpose)
def transpose(input, dim0, dim1):
    return ascend_op.Transpose(input, dim0, dim1)

@registe_conversion(torch.ops.aten.expand)
def expand(x, dims):
    return ascend_op.ExpandD(x, dims)

@registe_conversion(torch.ops.aten.view)
def view(x, shape):
    return ascend_op.TranShape(x, shape)

@registe_conversion(torch.ops.aten.bmm)
def bmm(x1, x2, adj_x1=False, adj_x2=False):
    return ascend_op.BatchMatMul(x1, x2, adj_x1, adj_x2)

@registe_conversion(torch.ops.aten.bernoulli.p)
def Bernoulli(x, p, generator=None):
    return ascend_op.Bernoulli(x, p, generator)

@registe_conversion(torch.ops.aten.new_empty_strided.default)
def NewEmptyStrided(x, size, stride, dtype = torch.float32, layout = torch.strided,
                      device = 'cpu', pin_memory = False):
    return ascend_op.NewEmptyStrided(x, size, stride, dtype, layout, device, pin_memory)

@registe_pattern
class ReplaceVarMean(BaseReplacePattern):
    def pattern(input, dims):
        return torch.ops.aten.var_mean.correction(input, dims, correction=0, keepdim=True)

    def replacement(input, dims):
        meanVal = torch.ops.aten.mean(input, dims, True)
        varVal = torch.ops.aten.var(input, dims, correction=1, keepdim=True)
        return ascend_op.ret_tuple(varVal, meanVal)

'''
@register_backend_pattern
class FuseTransposeBmm(BaseReplacePattern):
    def pattern(x1, x2):
        transpose_3 = transpose(x2, 2, 3)
        expand_1 = expand(transpose_3, [1, 32, 128, 32])
        view_16 = view(expand_1, [32, 128, 32])
        return bmm(x1, view_16)

    def replacement(x1, x2):
        view_16 = view(x2, [32, 32, 128])
        return bmm(x1, view_16, adj_x1 = False, adj_x2 = True)
'''
