import functools
import operator
import _operator
import torch
from typing import (
    Optional,
)
from torch.types import (
    Number,
)
import numpy as np
import torch.fx.traceback as fx_traceback
import dicp.vendor.AscendGraph.ascend_op as ascend_op
from dicp.dynamo_bridge.conversion import register_conversion_impl
from dicp.dynamo_bridge.op_transformer import SingleOpTransformer


aten = torch.ops.aten
prims = torch.ops.prims
conversions = {}

def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False

def register_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        register_conversion_impl,
        conversions,
        aten_fn,
    )

@register_conversion(torch.ops.aten.rsqrt)
def rsqrt(a):
    return ascend_op.Rsqrt(a)

@register_conversion(torch.ops.aten.sqrt)
def sqrt(a):
    return ascend_op.Sqrt(a)

@register_conversion(torch.ops.aten.log)
def log(a):
    return ascend_op.Log(a)

@register_conversion(torch.ops.aten.exp)
def exp(a):
    return ascend_op.Exp(a)

@register_conversion(torch.ops.aten.neg)
def neg(a):
    return ascend_op.Neg(a)

@register_conversion(torch.ops.aten.relu)
def relu(a):
    return ascend_op.Relu(a)

@register_conversion(torch.ops.aten.silu)
def silu(a):
    return ascend_op.Silu(a)

@register_conversion(torch.ops.aten._softmax)
def _softmax(x, dim, half_to_float):
    return ascend_op.Softmax(x, dim, half_to_float)

@register_conversion(torch.ops.aten.sum.default)
def sum(a):
    return ascend_op.Sum(a)

@register_conversion(torch.ops.aten.sum.dim_IntList)
def sumdim(x, dims, keepdim = True):
    return ascend_op.ReduceSumD(x, dims, keepdim)

@register_conversion(torch.ops.aten.clone)
def clone(a, memory_format = torch.contiguous_format):
    return ascend_op.Copy(a, memory_format)

@register_conversion(torch.ops.aten.copy_)
def copy_(dst, src):
    return ascend_op.Copy_(dst, src)

@register_conversion(torch.ops.aten.copy)
def copy(dst, src):
    return ascend_op.CopyInner(dst, src)

@register_conversion(torch.ops.prims.convert_element_type)
def convert_element_type(x, dtype):
    return ascend_op.Convert(x, dtype)

@register_conversion(torch.ops.aten.embedding.default)
def embedding(weight, indices, padding_idx=-1):
    return ascend_op.Embedding(weight, indices, padding_idx)

@register_conversion(torch.ops.aten.sigmoid)
def sigmoid(x):
    return ascend_op.Sigmoid(x)

@register_conversion(torch.ops.aten.pow)
def pow(x, exp):
    return ascend_op.Pow(x, exp)

@register_conversion(torch.ops.aten.ne)
def ne(x, scalar):
    return ascend_op.Ne(x, scalar)

@register_conversion(torch.ops.aten.unsqueeze)
def unsqueeze(x, dims):
    return ascend_op.Unsqueeze(x, dims)

@register_conversion(torch.ops.aten.squeeze)
def squeeze(x, dims):
    return ascend_op.Squeeze(x, dims)

@register_conversion(torch.ops.aten.permute)
def permute(x, dims):
    return ascend_op.Permute(x, dims)

@register_conversion(torch.ops.aten.cumsum.default)
def cumsum(x, dim, dtype=None):
    return ascend_op.CumSum(x, dim, dtype)

mm = torch.fx.wrap(register_conversion(torch.ops.aten.mm.default)(ascend_op.MatMul))


@register_conversion(torch.ops.aten.sort.default)
def sort(x, dim=-1, descending=False):
    return ascend_op.Sort(x, dim, descending)

@register_conversion(torch.ops.aten.topk.default)
def topk(x, k, dim=-1, largest=True, sorted=True):
    return ascend_op.TopK(x, k, dim, largest, sorted)

@register_conversion(torch.ops.aten.scatter.src)
def scatter(x, dims, index, value):
    return ascend_op.ScatterElement(x, dims, index, value)

@register_conversion(torch.ops.aten.mean)
def mean(x, dims=[], keepdim=False):
    return ascend_op.ReduceMean(x, dims, keepdim)

@register_conversion(torch.ops.aten.var)
def var(x, dims, correction, keepdim):
    return ascend_op.Var(x, dims, correction, keepdim)

@register_conversion(torch.ops.aten.amax)
def amax(x, dims, keepdim):
    return ascend_op.Amax(x, dims, keepdim)

@register_conversion(torch.ops.aten.gather)
def gather(x, dims, index):
    return ascend_op.GatherD(x, dims, index)

@register_conversion(torch.ops.aten.where)
def where(condition, a, b):
    return ascend_op.Where(condition, a, b)

@register_conversion(torch.ops.aten._unsafe_view)
def unsafe_view(x, shape):
    return ascend_op.TranShape(x, shape)

@register_conversion(_operator.mul)
def inmul(a, b):
    return ascend_op.InMul(a, b)

@register_conversion(_operator.ge)
def inge(a, b):
    return ascend_op.InGe(a, b)

@register_conversion(_operator.add)
def inadd(a, b):
    return ascend_op.InAdd(a, b)

@register_conversion(torch.ops.aten.sym_size)
def symsize(x, dim):
    return ascend_op.SymSize(x, dim)

@register_conversion(operator.getitem)
def identity(x, idx):
    return ascend_op.Identity(x, idx)

@register_conversion(torch.ops.aten.full_like)
def fulllike(x, value, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.FullLike(x, value, dtype, layout, device, pin_memory, memory_format)

@register_conversion(torch.ops.aten.lift_fresh_copy)
def lift_fresh_copy(tensor_constant):
    return ascend_op.LiftFreshCopy(tensor_constant)

@register_conversion(torch.ops.aten.lt.Scalar)
def lt(a, b):
    return ascend_op.Lt(a, b)

@register_conversion(torch.ops.aten.masked_fill.Scalar)
def masked_fill(x, mask, value):
    return ascend_op.MaskedFill(x, mask, value)

@register_conversion(torch.ops.aten.empty)
def empty(size, dtype=torch.int64, layout=torch.strided, device='cpu'):
    return ascend_op.Empty(size, dtype, layout, device)

@register_conversion(torch.ops.aten.index.Tensor)
def index(x, index):
    return ascend_op.Index(x, 0, index, None)

index_select = torch.fx.wrap(register_conversion(
    torch.ops.aten.index_select.default)(ascend_op.IndexSelect))

@register_conversion(torch.ops.aten.index_select.default)
def index_arg2_(x, index):
    return ascend_op.IndexSelect(x, 0, index, 2)

@register_conversion(torch.ops.aten.index_select.default)
def index_arg3_(x, dim, index):
    return ascend_op.IndexSelect(x, dim, index, 3)

@register_conversion(torch.ops.aten.fill.Scalar)
def fill(x, value):
    return ascend_op.Fill(x, value)

@register_conversion(torch.ops.aten.ones.default)
def ones(shape, dtype=torch.int64, device='cpu', pin_memory=False):
    return ascend_op.Ones(shape)

@register_conversion(torch.ops.aten.new_ones.default)
def new_ones(x, shape, dtype=torch.int64, layout=None, device='cpu', pin_memory=False):
    return ascend_op.NewOnes(x, shape)

@register_conversion(torch.ops.aten.repeat_interleave)
def repeat_interleave(repeats, output_size = 1):
    return ascend_op.RepeatInterleave(repeats, output_size)

@register_conversion(torch.ops.aten.full.default)
def full(dims, value, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.Full(dims, value, dtype, layout, device, pin_memory, memory_format)


@register_conversion(torch.ops.aten.max_pool2d_with_indices)
def maxpool2d(input, kernel_size, stride, padding):
    return ascend_op.MaxPool(input, kernel_size, stride, padding)

@register_conversion(torch.ops.aten.max_pool2d_with_indices_backward)
def maxpool2dbackward(grad, input, kernel_size, stride, padding, dilation, ceil_mode, index):
    return ascend_op.MaxPoolGradWithArgmaxV1(input, grad, index, kernel_size, stride, padding, dilation, ceil_mode)

@register_conversion(torch.torch.ops.aten.addmm)
def addmm(input, mat1, mat2):
    return ascend_op.AddMm(input, mat1, mat2)

@register_conversion(torch.ops.aten.convolution_backward)
def convolutionbackward(grad, input, weight, bias,
                stride, padding, dilation, transposed,
                output_padding, groups, output_masks):
    return ascend_op.ConvBackward(grad, input, weight, bias,
                stride, padding, dilation, transposed,
                output_padding, groups, output_masks)

@register_conversion(torch.ops.aten._log_softmax.default)
def log_softmax(x, dim, half_to_float):
    return ascend_op.LogSoftmax(x, dim, half_to_float)

@register_conversion(torch.ops.aten._log_softmax_backward_data.default)
def log_softmax_backward_data(grad_output, output, dim, input_dtype):
    return ascend_op.LogSoftmaxBackward(grad_output, output, dim, input_dtype)

@register_conversion(torch.ops.aten.nll_loss_forward.default)
def nll_loss_forward(x, target, weight, reduction, ignore_index):
    return ascend_op.NLLLossForward(x, target, weight, reduction, ignore_index)

@register_conversion(torch.ops.aten.nll_loss_backward.default)
def nll_loss_backward(grad_output, x, target, weight, reduction, ignore_index, total_weight):
    return ascend_op.NLLLossBackward(grad_output, x, target, weight, reduction,
                                     ignore_index, total_weight)

@register_conversion(torch.ops.aten._native_batch_norm_legit_functional.default)
def _native_batch_norm_legit_functional(x, weight, bias, running_mean, running_var,
                                        train, momentum, eps):
    return ascend_op.BatchNorm(x, weight, bias, running_mean, running_var, train,
                               momentum, eps)

@register_conversion(torch.ops.aten.threshold_backward.default)
def threshold_backward(grad_output, x, threshold):
    return ascend_op.ThresholdBackward(grad_output, x, threshold)

@register_conversion(torch.ops.aten.native_batch_norm_backward.default)
def native_batch_norm_backward(grad_out, x, weight, running_mean, running_var,
        save_mean, save_invstd, train, eps, grad_input_mask):
    return ascend_op.BatchNormBackward(grad_out, x, weight, running_mean, running_var,
            save_mean, save_invstd, train, eps, grad_input_mask)

@register_conversion(torch.ops.aten.zeros_like.default)
def zeros_like(x, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.ZerosLike(x)

@register_conversion(torch.ops.aten.slice.Tensor)
def slice(x, dim=0, start=None, end=None, step=1):
    return ascend_op.Slice(x, dim, start, end, step)

@register_conversion(torch.ops.aten.stack)
def stack(x, dim):
    return ascend_op.Stack(x, dim)

@register_conversion(torch.ops.aten.cat.default)
def cat(x, dim=0):
    return ascend_op.Cat(x, dim)

@register_conversion(torch.ops.aten.select.int)
def select(x, dim, index):
    return ascend_op.Select(x, dim, index)

@register_conversion(torch.ops.aten.lt.Tensor)
def lt(x, y):
    return ascend_op.Lt(x, y)
  
@register_conversion(torch.ops.aten.masked_fill.Scalar)
def masked_fill(x, y, value):
    return ascend_op.MaskedFill(x, y, value)

@register_conversion(torch.ops.aten.index.Tensor)
def index(*args, **kwargs):
    return ascend_op.Index(*args, **kwargs)

@register_conversion(torch.ops.aten._unsafe_view.default)
def unsafe_view(a, b):
    return ascend_op.UnsafeView(a, b)
 
@register_conversion(torch.ops.aten.slice_backward.default)
def slice_backward(grad, input_shape, dim, start, end, step):
    return ascend_op.SliceBackward(grad, input_shape, dim, start, end, step)

@register_conversion(torch.ops.aten.empty_like.default)
def empty_like(x, dtype = torch.float32, layout = torch.strided,
             device = 'cpu', pin_memory = False, memory_format = torch.preserve_format):
    return ascend_op.EmptyLike(x, dtype, layout, device, pin_memory, memory_format)
  
@register_conversion(torch.ops.aten.fill.Scalar)
def fill_scalar(x, value):
    return ascend_op.FillScalar(x, value)

@register_conversion(torch.ops.aten._softmax_backward_data.default)
def softmax_backward_data(grad_output, output, dim, input_dtype):
    return ascend_op.SoftmaxBackward(grad_output, output, dim, input_dtype)

@register_conversion(torch.ops.aten.lift_fresh_copy.default)
def LiftFreshCopy(*args, **kwargs):
    return ascend_op.LiftFreshCopy(*args, **kwargs)

@register_conversion(torch.ops.aten.eq.Tensor)
def Eq(x, y):
    return ascend_op.Eq(x, y)


class AtenToAscendTransformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)

    def process_dynamic_shape(self, shape):
        x_names = []
        def generate_digits_op(shapes):
            const_op = self.get_proxy(ascend_op.Const, (shapes, torch.int32), {})
            x_names.append(const_op)

        def generate_sym_int(elem):
            elem = elem.node.str()
            elems = elem.strip().split(' ')
            if len(elems) > 1:
                assert len(elems) == 3
                assert elems[2].isdigit()
                assert elems[1] == '+' or elems[1] == '-'
                const_op = self.get_proxy(ascend_op.Const, (int(elems[2]), torch.int32), {})
                args = (self.sym_to_inputs[elems[0]], const_op)
                if elems[1] == '+':
                    x_names.append(self.get_proxy(ascend_op.Add, args, {}))
                else:
                    x_names.append(self.get_proxy(ascend_op.Sub, args, {}))
            else:
                x_names.append(self.sym_to_inputs[elems[0]])

        dims = []
        for elem in shape:
            if not isinstance(elem, torch.SymInt):
                dims.append(elem)
                continue
            st = elem.node.str()
            if st.isdigit():
                dims.append(int(st))
                continue

            if len(dims) > 0:
                generate_digits_op(dims)
                dims = []
            generate_sym_int(elem) 
        if len(dims) > 0:
            generate_digits_op(dims)
        # concat all ops
        return self.get_proxy(ascend_op.ConcatD, (x_names, 0), {})

    def mul_scalar(self, x, y):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        const_dtype = torch.float32 if out_dtype == torch.float16 else out_dtype
        y_op = self.get_proxy(ascend_op.Const, (y, const_dtype), {})
        y_shape = list(x.node.meta['val'].shape)
        if symint_in_shape(y_shape):
            y_shape_op = self.process_dynamic_shape(y_shape)
            y_op = self.get_proxy(ascend_op.BroadCast, (y_op, y_shape_op))
        if out_dtype == torch.float16:
            y_op = self.get_proxy(ascend_op.Cast, (y_op, torch.float16), {})
        return self.get_proxy(ascend_op.Mul, (x, y_op), {})

    def mul_complex64(self, x, y):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        assert out_dtype == torch.complex64
         # (a + bj)*(c + dj) = (ac - bd)+(ad + bc)j
        a = self.get_proxy(ascend_op.Identity, (x, 0), {})
        b = self.get_proxy(ascend_op.Identity, (x, 1), {})
        c = self.get_proxy(ascend_op.Identity, (y, 0), {})
        d = self.get_proxy(ascend_op.Identity, (y, 1), {})

        ac = self.get_proxy(ascend_op.Mul, (a, c), {})
        bd = self.get_proxy(ascend_op.Mul, (b, d), {})
        ad = self.get_proxy(ascend_op.Mul, (a, d), {})
        bc = self.get_proxy(ascend_op.Mul, (b, c), {})

        ac_bd = self.get_proxy(ascend_op.Sub, (ac, bd), {})
        ad_bc = self.get_proxy(ascend_op.Add, (ad, bc), {})

        out = self.get_proxy(ascend_op.IdentityN, (ac_bd, ad_bc), {})
        return out

    @register_conversion(torch.ops.aten.mul)
    def mul(self, x, y):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        if out_dtype == torch.complex64:
            return self.mul_complex64(x, y)
        if not isinstance(y, torch.fx.proxy.Proxy):
            return self.mul_scalar(x, y)
        x_shape = list(x.node.meta['val'].shape)
        y_shape = list(y.node.meta['val'].shape)  
        x_dtype = x.node.meta['val'].dtype
        y_dtype = y.node.meta['val'].dtype
        # handling with broadcasting cases
        if np.prod(x_shape) < np.prod(y_shape):
            if symint_in_shape(y_shape):
                y_shape_op = self.process_dynamic_shape(y_shape)
                x = self.get_proxy(ascend_op.BroadCast, (x, y_shape_op))
        elif np.prod(x_shape) > np.prod(y_shape):
            if symint_in_shape(x_shape):
                x_shape_op = self.process_dynamic_shape(x_shape)
                y = self.get_proxy(ascend_op.BroadCast, (y, x_shape_op))
        if x_dtype != out_dtype:
            x = self.get_proxy(ascend_op.Cast, (x, out_dtype), {})
        if y_dtype != out_dtype:
            y = self.get_proxy(ascend_op.Cast, (y, out_dtype), {})
        return self.get_proxy(ascend_op.Mul, (x, y), {})
    
    @register_conversion(torch.ops.aten.add.Tensor)
    def add(self, x, y, alpha: Optional[Number] = 1):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        if not isinstance(y, torch.fx.proxy.Proxy):
            y = y * alpha
            if out_dtype == torch.float or out_dtype == torch.float16:
                return self.get_proxy(ascend_op.Adds, (x, float(y)), {})
            else:
                y = self.get_proxy(ascend_op.Const, (y, out_dtype), {})
        else:
            x_dtype = x.node.meta['val'].dtype
            y_dtype = y.node.meta['val'].dtype
            y = self.mul(y, alpha)
            if x_dtype != out_dtype:
                x = self.get_proxy(ascend_op.Cast, (x, out_dtype), {})
            if y_dtype != out_dtype:
                y = self.get_proxy(ascend_op.Cast, (y, out_dtype), {})
        return self.get_proxy(ascend_op.Add, (x, y), {})
    
    @register_conversion(torch.ops.aten._to_copy.default)
    def _to_copy(self, x, dtype=None, layout=torch.strided, device='cpu'):
        if dtype:
            return self.get_proxy(ascend_op.Cast, (x, dtype), {})
        else:
            return self.get_proxy(ascend_op.Identity, (x), {})
    
    @register_conversion(aten.le)
    def le(self, a, b):
        if isinstance(b, torch.fx.proxy.Proxy):
            return self.get_proxy(ascend_op.LessEqual, (a, b), {})
        x2 = self.get_proxy(ascend_op.Const, (b, torch.float32), {})
        if a.node.meta['val'].dtype == torch.float16:
            x2 = self.get_proxy(ascend_op.Cast, (x2, torch.float16), {})
        return self.get_proxy(ascend_op.LessEqual, (a, x2), {})

    @register_conversion(aten.view_as_real)
    def view_as_real(self, x):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        assert out_dtype == torch.float32
        x_shape = list(x.node.meta['val'].shape)
        dim = len(x_shape)
        op1 = self.get_proxy(ascend_op.Identity, (x, 0), {})
        op2 = self.get_proxy(ascend_op.Identity, (x, 1), {})
        pack = self.get_proxy(ascend_op.Pack, ([op1, op2], dim), {})
        return self.get_proxy(ascend_op.Squeeze, (pack, [-1]), {})


@register_conversion(torch.ops.aten.bernoulli.p)
def Bernoulli(x, p, generator=None):
    return ascend_op.Bernoulli(x, p, generator)

@register_conversion(torch.ops.aten.new_empty_strided.default)
def NewEmptyStrided(x, size, stride, dtype = torch.float32, layout = torch.strided,
                      device = 'cpu', pin_memory = False):
    return ascend_op.NewEmptyStrided(x, size, stride, dtype, layout, device, pin_memory)

arange = torch.fx.wrap(register_conversion(aten.arange.default)(ascend_op.Arange))
eq = torch.fx.wrap(register_conversion(aten.eq)(ascend_op.Eq))
div = torch.fx.wrap(register_conversion(aten.div)(ascend_op.Div))
maximum = torch.fx.wrap(register_conversion(aten.maximum.default)(ascend_op.Max))
convolution = torch.fx.wrap(register_conversion(aten.convolution)(ascend_op.Conv2D))
abs = torch.fx.wrap(register_conversion(aten.abs)(ascend_op.Abs))
t = torch.fx.wrap(register_conversion(aten.t.default)(ascend_op.T))
transpose = torch.fx.wrap(register_conversion(aten.transpose.int)(ascend_op.Transpose))
expand = torch.fx.wrap(register_conversion(aten.expand.default)(ascend_op.ExpandD))
view = torch.fx.wrap(register_conversion(aten.view.default)(ascend_op.TranShape))
bmm = torch.fx.wrap(register_conversion(aten.bmm.default)(ascend_op.BatchMatMul))
sub = torch.fx.wrap(register_conversion(aten.sub)(ascend_op.Sub))
rsub = torch.fx.wrap(register_conversion(aten.rsub)(ascend_op.Rsub))
view_as_complex = torch.fx.wrap(register_conversion(aten.view_as_complex.default)(ascend_op.ViewAsComplex))
