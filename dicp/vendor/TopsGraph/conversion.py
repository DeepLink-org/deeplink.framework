import torch
import functools
from . import tops_op
import numbers
import torch.fx.traceback as fx_traceback
from torch.fx import Proxy
import operator
from dicp.dynamo_bridge.op_transformer import SingleOpTransformer
from dicp.dynamo_bridge.compile_fx import is_torch_210
from typing import (
    Optional,
)
from torch.types import (
    Number,
)
from dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)

conversions = {}
patterns = []
aten = torch.ops.aten
prims = torch.ops.prims


def args_kwargs_unchange(args, kwargs):
    return args, kwargs


def _register_conversion(
    aten_fn, decomp_fn, process_args_kwargs_fn=None
):
    register_op_singleton_flag = isinstance(
        decomp_fn, type) and issubclass(decomp_fn, tops_op.Operator)
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
                if other_fn not in conversions:
                    aten_fn.append(other_fn)

    conversions.update({fn: wrapped for fn in aten_fn})
    if register_op_singleton_flag:
        return wrapped[0]
    else:
        return wrapped


def register_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_conversion,
        aten_fn,
    )


class AtenToTopsTransformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)

    @register_conversion(aten.add.Tensor)
    def Add(self, x, y, alpha: Optional[Number] = 1):
        y_node = y.node if isinstance(y, torch.fx.proxy.Proxy) else y
        try:
            in_dtype = x.node.meta["val"].dtype
            out_dtype = fx_traceback.get_current_meta()['val'].dtype
            if in_dtype != out_dtype:
                x = self.get_proxy(tops_op.Convert, (x, out_dtype))
        except Exception:
            pass
        if not isinstance(y_node, torch.fx.node.Node):
            y = y * alpha
        elif alpha != 1:
            y = self.get_proxy(tops_op.Mul, (y, alpha))
        return self.get_proxy(tops_op.Add, (x, y))

    @register_conversion(aten.abs)
    def Abs(self, *args, **kwargs):
        return self.get_proxy(tops_op.Abs, args, kwargs)

    @register_conversion(aten.add.default)
    def AddDefalut(self, *args, **kwargs):
        return self.get_proxy(tops_op.AddDefalut, args, kwargs)

    @register_conversion(aten.add.Scalar)
    def AddScalar(self, *args, **kwargs):
        return self.get_proxy(tops_op.AddScalar, args, kwargs)

    @register_conversion(aten.mul)
    def Mul(self, a, b):
        if isinstance(a, Proxy):
            if hasattr(a.node, "meta") and 'val' in a.node.meta:
                if (a.node.meta['val'].dtype == torch.complex64) or (a.node.meta['val'].dtype == torch.cfloat):
                    return tops_op.ComplexMul(a, b)
        return tops_op.Mul(a, b)

    @register_conversion(aten.mul.Scalar)
    def MulScalar(self, *args, **kwargs):
        return self.get_proxy(tops_op.MulScalar, args, kwargs)

    @register_conversion(aten.div)
    def Div(self, a, b):
        a_node = a.node if isinstance(a, torch.fx.proxy.Proxy) else a
        in_dtype = a_node.meta["val"].dtype
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        if in_dtype is torch.float16 or out_dtype is torch.float16:
            a = self.get_proxy(tops_op.Convert, (a, torch.float32))
            if not isinstance(b, numbers.Number):
                b = self.get_proxy(tops_op.Convert, (b, torch.float32))
            res = self.get_proxy(tops_op.Div, (a, b))
            return self.get_proxy(tops_op.Convert, (res, torch.float16))
        return self.get_proxy(tops_op.Div, (a, b))

    @register_conversion(aten.sub)
    def Sub(self, *args, **kwargs):
        return self.get_proxy(tops_op.Sub, args, kwargs)

    @register_conversion(aten.sqrt)
    def Sqrt(self, *args, **kwargs):
        return self.get_proxy(tops_op.Sqrt, args, kwargs)

    @register_conversion(aten.reciprocal)
    def Reciprocal(self, *args, **kwargs):
        return self.get_proxy(tops_op.Reciprocal, args, kwargs)

    @register_conversion(aten.rsqrt)
    def Rsqrt(self, *args, **kwargs):
        return self.get_proxy(tops_op.Rsqrt, args, kwargs)

    @register_conversion(aten.exp)
    def Exp(self, *args, **kwargs):
        return self.get_proxy(tops_op.Exp, args, kwargs)

    @register_conversion(aten.relu)
    def Relu(self, *args, **kwargs):
        return self.get_proxy(tops_op.Relu, args, kwargs)

    @register_conversion(aten.sum)
    def ReduceSum(self, a, *args, **kwargs):
        if isinstance(a, Proxy):
            if hasattr(a.node, "meta"):
                in_dtype = a.node.meta["val"].dtype
                out_dtype = fx_traceback.get_current_meta()['val'].dtype
                if in_dtype != out_dtype:
                    a = self.get_proxy(tops_op.Convert, (a, out_dtype))
        return self.get_proxy(tops_op.ReduceSum, (a, *args), kwargs)

    @register_conversion(operator.getitem)
    def GetTupleElement(self, a, dim, **kwargs):
        dim = dim % len(a.node.meta["val"])
        return self.get_proxy(tops_op.GetTupleElement, (a, dim), kwargs)

    @register_conversion(aten.index.Tensor)
    def Index(self, *args, **kwargs):
        assert len(args[1]) == 1, "Only support aten.index with one index arg"
        idx_rank = len(args[1][0].node.meta['val'].shape)
        slice_size = list(args[0].node.meta['val'].shape)
        slice_size[0] = 1
        return self.get_proxy(tops_op.XlaGather, (args[0], args[1][0],
                              [idx_rank,], [0,], [0,], idx_rank, [1, args[0].node.meta['val'].shape[1]]))

    # tops_dropout only returns a tensor, not a tuple of tensor
    @register_conversion(aten.native_dropout.default)
    def NativeDropout(self, *args, **kwargs):
        dropout = self.get_proxy(tops_op.NativeDropout, args)
        data_type = args[0].node.meta["val"].dtype
        ne = self.get_proxy(tops_op.NotEqual, (data_type, dropout, 0))
        return self.get_proxy(tops_op.MakeTuple, (dropout, ne))

    @register_conversion(aten.squeeze)
    def Squeeze(self, *args, **kwargs):
        return self.get_proxy(tops_op.Squeeze, args, kwargs)

    @register_conversion(aten.unsqueeze)
    def Unsqueeze(self, *args, **kwargs):
        return self.get_proxy(tops_op.Unsqueeze, args, kwargs)

    @register_conversion(aten.permute)
    def Permute(self, *args, **kwargs):
        return self.get_proxy(tops_op.Transpose, args, kwargs)

    @register_conversion(aten.transpose)
    def Transpose(self, *args, **kwargs):
        return self.get_proxy(tops_op.Transpose1, args, kwargs)

    @register_conversion(aten.hardswish)
    def Hardswish(self, *args, **kwargs):
        return self.get_proxy(tops_op.Hardswish, args, kwargs)

    @register_conversion(aten.hardswish_backward)
    def HardswishBackward(self, *args, **kwargs):
        return self.get_proxy(tops_op.HardswishBackward, args, kwargs)

    @register_conversion(aten.clone)
    def Clone(self, *args, **kwargs):
        return self.get_proxy(tops_op.Clone, args, kwargs)

    @register_conversion(aten.copy.default)
    def Copy(self, *args, **kwargs):
        return self.get_proxy(tops_op.Copy, args, kwargs)

    @register_conversion(aten.copy_.default)
    def Copy_(self, *args, **kwargs):
        return self.get_proxy(tops_op.Copy, args, kwargs)

    @register_conversion(aten.lift_fresh_copy.default)
    def LiftFreshCopy(self, *args, **kwargs):
        return self.get_proxy(tops_op.LiftFreshCopy, args, kwargs)

    @register_conversion(aten.alias)
    def Alias(self, *args, **kwargs):
        return self.get_proxy(tops_op.Alias, args, kwargs)

    @register_conversion(aten.neg)
    def Neg(self, *args, **kwargs):
        return self.get_proxy(tops_op.Neg, args, kwargs)

    @register_conversion(aten.mean)
    def ReduceMean(self, a, dim=None, keepdim=False, **kwargs):
        in_shape = a.node.meta["val"].shape
        if dim is None:
            dim = list(range(len(in_shape)))
            return self.get_proxy(tops_op.ReduceMean, (a, dim))
        dim = [(item + len(in_shape)) if item < 0 else item for item in dim]
        return self.get_proxy(tops_op.ReduceMean, (a, dim, keepdim))

    @register_conversion(aten.lt.Tensor)
    def Less(self, *args, **kwargs):
        return self.get_proxy(tops_op.Less, args, kwargs)

    @register_conversion(aten.le.Scalar)
    def LessEqual(self, *args, **kwargs):
        return self.get_proxy(tops_op.LessEqual, args, kwargs)

    @register_conversion(aten.eq.Tensor)
    def Equal(self, *args, **kwargs):
        return self.get_proxy(tops_op.Equal, args, kwargs)

    @register_conversion(aten.eq.Scalar)
    def EqualScalar(self, *args, **kwargs):
        return self.get_proxy(tops_op.EqualScalar, args, kwargs)

    @register_conversion(aten.ne.Scalar)
    def NotEqual(self, a, b):
        data_type = a.node.meta["val"].dtype
        return self.get_proxy(tops_op.NotEqual, (data_type, a, b))

    @register_conversion(aten.view)
    def Reshape(self, *args, **kwargs):
        if args[0].node.meta["val"].dtype in (torch.cfloat, torch.cdouble):
            x = self.get_proxy(tops_op.GetTupleElement, (args[0], 0))
            x = self.get_proxy(tops_op.Reshape, (x, *args[1:]), kwargs)
            y = self.get_proxy(tops_op.GetTupleElement, (args[0], 1))
            y = self.get_proxy(tops_op.Reshape, (y, *args[1:]), kwargs)
            return self.get_proxy(tops_op.MakeTuple, (x, y))
        return self.get_proxy(tops_op.Reshape, args, kwargs)

    @register_conversion(aten.convolution)
    def Convolution(self, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
        inputs = [item for item in (x, weight, bias) if item is not None]
        padding = [padding[0], padding[0]] if len(padding) == 1 else list(padding)
        return self.get_proxy(tops_op.Convolution, (inputs, x, weight, bias, stride, padding, dilation,
                                                    transposed, output_padding, groups))

    @register_conversion(aten.convolution_backward.default)
    def ConvolutionBackward(self, grad_output, a, weight, bias_size, stride, padding, dilation, *args, **kwargs):
        inputs = [item for item in (grad_output, a, weight)]
        return self.get_proxy(tops_op.ConvolutionBackward, (inputs, grad_output, a, weight, bias_size,
                                                            stride, padding, dilation, *args), kwargs)

    @register_conversion(aten.max_pool2d_with_indices)
    def Max_pool2d_with_indices(self, x, kernel_size, stride=[], padding=[0, 0], dilation=[1, 1], ceil_mode=False):
        out_shape = fx_traceback.get_current_meta()["val"][0].shape
        return self.get_proxy(tops_op.Max_pool2d_with_indices, (out_shape, x, kernel_size, stride, padding, dilation, ceil_mode))

    @register_conversion(aten.max_pool2d_with_indices_backward)
    def MaxPool2DBackward(self, *args, **kwargs):
        return self.get_proxy(tops_op.Max_pool2d_with_indices_backward, args, kwargs)

    @register_conversion(aten._adaptive_avg_pool2d.default)
    def Adaptive_avg_pool2d(self, *args, **kwargs):
        assert len(args) == 2 and args[1] == [1, 1], "limited support"
        reudce_dim = [2, 3]
        return self.get_proxy(tops_op.Adaptive_avg_pool2d, (reudce_dim, *args), kwargs)

    @register_conversion(aten._adaptive_avg_pool2d_backward.default)
    def Adaptive_avg_pool2d_backward(self, grad_output, inputs):
        out_shape = fx_traceback.get_current_meta()["val"].shape
        expand = self.get_proxy(tops_op.Expand, (grad_output, out_shape))
        value = out_shape[2] * out_shape[3]
        scalar = self.get_proxy(tops_op.Scalar, (value, ))
        return self.get_proxy(tops_op.Div, (expand, scalar))

    @register_conversion(aten.gather)
    def Gather(self, a, dim, index, *args, **kwargs):
        in_shape = a.node.meta["val"].shape
        dim = dim % len(in_shape)
        return self.get_proxy(tops_op.Gather, (a, dim, index, *args), kwargs)

    @register_conversion(aten.log)
    def Log(self, *args, **kwargs):
        return self.get_proxy(tops_op.Log, args, kwargs)

    @register_conversion(aten.amax)
    def ReduceMax(self, *args, **kwargs):
        return self.get_proxy(tops_op.ReduceMax, args, kwargs)

    @register_conversion(aten._native_batch_norm_legit_functional.default)
    def BatchNorm(self, *args, **kwargs):
        return self.get_proxy(tops_op.BatchNorm, args, kwargs)

    @register_conversion(aten.native_batch_norm_backward.default)
    def BatchNormBackward(*args, **kwargs):
        return tops_op.BatchNormBackward(*args, **kwargs)

    @register_conversion(aten._softmax)
    def Softmax(self, a, dim, half_to_float):
        out_shape = fx_traceback.get_current_meta()["val"].shape
        dim = dim + len(out_shape) if dim < 0 else dim
        return self.get_proxy(tops_op.Softmax, (a, dim, half_to_float))

    @register_conversion(aten.dot.default)
    def Dot(self, *args, **kwargs):
        return self.get_proxy(tops_op.Dot, args, kwargs)

    @register_conversion(aten.mm)
    def Gemm(self, *args, **kwargs):
        return self.get_proxy(tops_op.Gemm, args)

    @register_conversion(aten.bmm.default)
    def Bmm(self, *args, **kwargs):
        return self.get_proxy(tops_op.DotGeneral, (*args, [0,], [0,], [2,], [1,]))

    @register_conversion(aten.cat.default)
    def Concatenate(self, *args, **kwargs):
        tensors = []
        for arg in args[0]:
            if torch.numel(arg.node.meta['val']):
                tensors.append(arg)
        dim = 0 if len(args) < 2 else args[1]
        dim = dim % len(args[0][0].node.meta["val"].shape)
        return self.get_proxy(tops_op.Concatenate, (args[0], dim))

    @register_conversion(aten.empty_like.default)
    def EmptyLike(self, *args, **kwargs):
        return self.get_proxy(tops_op.EmptyLike, args, kwargs)

    @register_conversion(aten.bernoulli.p)
    def Bernoulli(self, *args, **kwargs):
        return self.get_proxy(tops_op.Bernoulli, args, kwargs)

    @register_conversion(aten.new_empty_strided.default)
    def NewEmptyStrided(self, *args, **kwargs):
        return self.get_proxy(tops_op.NewEmptyStrided, args, kwargs)

    @register_conversion(aten.expand.default)
    def Expand(self, *args, **kwargs):
        return self.get_proxy(tops_op.Expand, args, kwargs)

    @register_conversion(aten.full.default)
    def Full(self, *args, **kwargs):
        return self.get_proxy(tops_op.Full, args, kwargs)

    @register_conversion(aten.full_like.default)
    def FullLike(self, *args, **kwargs):
        return self.get_proxy(tops_op.FullLike, args, kwargs)

    @register_conversion(aten.maximum.default)
    def Max(self, *args, **kwargs):
        return self.get_proxy(tops_op.Max, args, kwargs)

    @register_conversion(aten.pow.Tensor_Scalar)
    def Pow(self, *args, **kwargs):
        return self.get_proxy(tops_op.Pow, args, kwargs)

    @register_conversion(aten.sigmoid.default)
    def Sigmoid(self, *args, **kwargs):
        return self.get_proxy(tops_op.Sigmoid, args, kwargs)

    @register_conversion(aten.slice.Tensor)
    def Slice(self, a, dim=0, start=0, end=-1, step=1, **kwargs):
        if isinstance(a, Proxy):
            if hasattr(a.node, "meta"):
                in_shape = a.node.meta["val"].shape
                out_shape = fx_traceback.get_current_meta()["val"].shape
                if in_shape != out_shape:
                    start = start % in_shape[dim]
                    end = end + in_shape[dim] if end < 0 else end
                    end = in_shape[dim] if end > in_shape[dim] else end
                    return self.get_proxy(tops_op.SliceInDim, (a, dim, start, end, step), kwargs)
                start_indices = [0 for _ in range(len(out_shape))]
                limit_indices = in_shape
                strides = [1 for _ in range(len(out_shape))]
        return self.get_proxy(tops_op.Slice, (start_indices, limit_indices, strides, a, dim, start, end, step), kwargs)

    @register_conversion(aten.slice_scatter.default)
    def SliceScatter(self, a, b, dim=0, start=0, end=-1, step=1):
        if isinstance(a, Proxy):
            if hasattr(a.node, "meta"):
                operand_shape = a.node.meta["val"].shape
                end = end % operand_shape[dim] if end < operand_shape[dim] else operand_shape[dim]
                assert end == operand_shape[dim] and step == 1, "limited support"
                return self.get_proxy(tops_op.SliceScatter, (a, b, dim, start, end, step))

    @register_conversion(aten.select.int)
    def Select(self, a, dim, index):
        if isinstance(a, Proxy):
            if hasattr(a.node, "meta"):
                in_shape = a.node.meta["val"].shape
                index = index % in_shape[dim]
                slice = self.get_proxy(
                    tops_op.SliceInDim, (a, dim, index, index + 1, 1))
                return self.get_proxy(tops_op.Squeeze, (slice, dim))

    @register_conversion(aten.where.self)
    def Where(self, *args, **kwargs):
        return self.get_proxy(tops_op.Where, args, kwargs)

    @register_conversion(aten.scatter.value)
    def Scatter(self, *args, **kwargs):
        return self.get_proxy(tops_op.Scatter, args, kwargs)

    @register_conversion(aten.zeros_like)
    def ZerosLike(self, *args, **kwargs):
        return self.get_proxy(tops_op.ZerosLike, args, kwargs)

    @register_conversion(aten.ones_like)
    def OnesLike(self, *args, **kwargs):
        return self.get_proxy(tops_op.OnesLike, args, kwargs)

    @register_conversion(aten.scalar_tensor.default)
    def Scalar(self, a, **kwargs):
        if "dtype" in kwargs:
            real_dtype = kwargs["dtype"]
            if real_dtype not in (torch.int64, torch.float32):
                kwargs["dtype"] = torch.float32
                scalar = self.get_proxy(tops_op.Scalar, (a,), kwargs)
                return self.get_proxy(tops_op.Convert(), (scalar, real_dtype))
        return self.get_proxy(tops_op.Scalar, (a,), kwargs)

    @register_conversion(aten.embedding)
    def Embedding(self, *args, **kwargs):
        idx_rank = len(args[1].node.meta['val'].shape)
        return self.get_proxy(tops_op.XlaGather, (*args,
                                                  [idx_rank,], [0,], [0,], idx_rank, [1, args[0].node.meta['val'].shape[1]]))

    @register_conversion(prims.convert_element_type)
    def Convert(self, *args, **kwargs):
        return self.get_proxy(tops_op.Convert, args, kwargs)

    @register_conversion(aten.view_as_complex)
    def ViewAsComplex(self, *args, **kwargs):
        return self.get_proxy(tops_op.ViewAsComplex, args, kwargs)

    @register_conversion(aten.view_as_real)
    def ViewAsReal(self, *args, **kwargs):
        return self.get_proxy(tops_op.ViewAsReal, args, kwargs)

    @register_conversion(aten._unsafe_view.default)
    def UnsafeView(self, *args, **kwargs):
        return self.get_proxy(tops_op.UnsafeView, args, kwargs)

    @register_conversion(aten._log_softmax.default)
    def Logsoftmax(self, *args, **kwargs):
        return self.get_proxy(tops_op.Logsoftmax, args, kwargs)

    @register_conversion(aten.gelu.default)
    def Gelu(self, *args, **kwargs):
        approximate = 'true' if ('approximate' in kwargs
                                 and kwargs["approximate"] == 'tanh') else 'false'
        return self.get_proxy(tops_op.Gelu, (args[0], approximate))

    @register_conversion(aten.gelu_backward.default)
    def gelubackward(self, *args, **kwargs):
        approximate = 'true' if ('approximate' in kwargs
                                 and kwargs["approximate"] == 'tanh') else 'false'
        return self.get_proxy(tops_op.GeluBackward, (args[0], args[1], approximate))

    @register_conversion(prims.iota.default)
    def Iota(self, length, **kwargs):
        iota = self.get_proxy(tops_op.Iota, (length,), kwargs)
        if kwargs["start"] != 0 or kwargs["step"] != 1:
            offset = self.get_proxy(tops_op.Mul, (iota, kwargs["step"]))
            return self.get_proxy(tops_op.Add, (offset, kwargs["start"]))
        return iota


# Patterns
tops_patterns = PatternMatcherPass()
aten_patterns_cls_list = []
register_aten_patterns = functools.partial(
    register_backend_patterns, aten_patterns_cls_list)
tops_patterns_cls_list = []
register_tops_patterns = functools.partial(
    register_backend_patterns, tops_patterns_cls_list)


@register_aten_patterns
class ReplacePatternAddmm(BackendPatternBase):
    @staticmethod
    def pattern(a, b, c):
        return torch.ops.aten.addmm.default(a, b, c)

    @staticmethod
    def replacement(a, b, c):
        return torch.ops.aten.add.Tensor(a, torch.ops.aten.mm(b, c))


# %var: [#users=2] = call_function[target=torch.ops.aten.var.correction]
#                                      (args = (%convolution_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
@register_aten_patterns
class ReplacePatternVar(BackendPatternBase):
    @staticmethod
    def pattern(a, b):
        return torch.ops.aten.var.correction(a, b, correction=0, keepdim=True)

    @staticmethod
    def replacement(inputs, dims):
        keepdim = True
        correction = 0
        denom = 64
        denom = denom - correction
        mean1 = torch.ops.aten.mean.dim(inputs, dims, keepdim)
        diffs = torch.ops.aten.square.default(
            torch.ops.aten.sub.Tensor(inputs, mean1))
        sum_results = torch.ops.aten.sum.dim_IntList(diffs, dims, keepdim)
        x_var = torch.ops.aten.div.Tensor(sum_results, denom)
        return x_var


# %var_mean_correction_4 : [#users=2] = call_function[target=torch.ops.aten.var_mean.correction]
#                                      (args = (%convolution_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
@register_aten_patterns
class ReplacePatternVarMean(BackendPatternBase):
    @staticmethod
    def pattern(a, b):
        return torch.ops.aten.var_mean.correction(a, b, correction=0, keepdim=True)

    @staticmethod
    def replacement(inputs, dims):
        keepdim = True
        correction = 0
        denom = 64
        denom = denom - correction
        mean1 = torch.ops.aten.mean.dim(inputs, dims, keepdim)
        diffs = torch.ops.aten.square.default(
            torch.ops.aten.sub.Tensor(inputs, mean1))
        sum_results = torch.ops.aten.sum.dim_IntList(diffs, dims, keepdim)
        x_var = torch.ops.aten.div.Tensor(sum_results, denom)
        return tops_op.ret_tuples(x_var, mean1)


@register_aten_patterns
class ReplacePatternT(BackendPatternBase):
    @staticmethod
    def pattern(a):
        return torch.ops.aten.t.default(a)

    @staticmethod
    def replacement(inputs):
        return torch.ops.aten.transpose(inputs, 0, 1)


@register_aten_patterns
class ReplacePatternRsub(BackendPatternBase):
    @staticmethod
    def pattern(a, b):
        return torch.ops.aten.rsub.Scalar(a, b)

    @staticmethod
    def replacement(a, b):
        return torch.ops.aten.sub.Scalar(b, a)


@register_aten_patterns
class ReplacePatternSiLU(BackendPatternBase):
    # silu(x) = x / (1+exp(-x)) = x*sigmoid(x)
    @staticmethod
    def pattern(a):
        return torch.ops.aten.silu.default(a)

    @staticmethod
    def replacement(a):
        return torch.ops.aten.mul.default(a, torch.ops.aten.sigmoid.default(a))


if is_torch_210:
    Gemm = torch.fx.wrap(tops_op.Gemm.get_singleton())
    DotGeneral = torch.fx.wrap(tops_op.DotGeneral.get_singleton())
    Permute = torch.fx.wrap(tops_op.Transpose.get_singleton())
    Transpose = torch.fx.wrap(tops_op.Transpose1.get_singleton())
    Expand = torch.fx.wrap(tops_op.Expand.get_singleton())
    Reshape = torch.fx.wrap(tops_op.Reshape.get_singleton())
    Bmm = torch.fx.wrap(tops_op.Bmm.get_singleton())

    @register_tops_patterns
    class GemmTransposeRhsPattern(BackendPatternBase):
        @staticmethod
        def pattern(reshaped_input, weight):
            transposed_weight = Permute(weight, [1, 0])
            return Gemm(reshaped_input, transposed_weight)

        @staticmethod
        def replacement(reshaped_input, weight):
            return DotGeneral(reshaped_input, weight, [], [], [1,], [1,])

    @register_tops_patterns
    class LlamaMatmulTransposePattern(BackendPatternBase):
        @staticmethod
        def pattern(xq, keys, expanded_xq_size, reshaped_xq_size, expanded_keys_size, reshaped_keys_size):
            xq_1 = Permute(xq, [0, 2, 1, 3])
            keys_1 = Permute(keys, [0, 2, 1, 3])
            keys_2 = Permute(keys_1, [0, 1, 3, 2])
            expanded_xq = Expand(xq_1, expanded_xq_size)
            reshaped_xq = Reshape(expanded_xq, reshaped_xq_size)
            expanded_keys = Expand(keys_2, expanded_keys_size)
            reshaped_keys = Reshape(expanded_keys, reshaped_keys_size)
            bmm_res = Bmm(reshaped_xq, reshaped_keys)
            return bmm_res

        @staticmethod
        def replacement(xq, keys):
            return DotGeneral(xq, keys, [0, 2], [0, 2], [3,], [3,])
