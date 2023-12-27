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
from torch._subclasses import FakeTensor
import dicp.vendor.AscendGraph.ascend_op as ascend_op
from dicp.vendor.AscendGraph.codegen.utils import (
    symint_in_shape,
    get_ascend_dtype,
    get_cpp_dtype
)
from dicp.dynamo_bridge.conversion import register_conversion_impl
from dicp.dynamo_bridge.op_transformer import SingleOpTransformer


aten = torch.ops.aten
prims = torch.ops.prims
conversions = {}


def get_reduction_str(r):
    if r == 0:
        return "none"
    elif r == 1:
        return "mean"
    elif r == 2:
        return "sum"
    else:
        raise RuntimeError("not supported yet!")


def register_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        register_conversion_impl,
        conversions,
        aten_fn,
    )


class AtenToAscendTransformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)

    def process_dynamic_shape(self, shape):
        x_names = []

        def generate_digits_op(shapes):
            const_op = self.get_proxy(
                ascend_op.Const, (shapes, torch.int32, [len(shapes)]))
            x_names.append(const_op)

        def generate_sym_int(elem):
            elem = elem.node.str()
            elems = elem.strip().split(' ')

            arg = None
            # dynamic shape feature
            if elems[0] in self.sym_in_args:
                arg, idx = self.sym_in_args[elems[0]]
                shape = self.get_proxy(ascend_op.Shape, (arg,))
                axis = self.get_proxy(
                    ascend_op.Const, ([0], torch.int32, [1]))
                indice = self.get_proxy(
                    ascend_op.Const, ([idx], torch.int32, [1]))
                gather = self.get_proxy(
                    ascend_op.GatherV2, (shape, indice, axis))

            if len(elems) > 1:
                assert len(elems) == 3
                assert elems[2].isdigit()
                assert elems[1] == '+' or elems[1] == '-'
                const_op = self.get_proxy(
                    ascend_op.Const, ([int(elems[2])], torch.int32, [1]))
                if arg is not None:
                    args = (gather, const_op)
                else:
                    args = (self.sym_to_inputs[elems[0]], const_op)
                if elems[1] == '+':
                    x_names.append(self.get_proxy(ascend_op.Add, args))
                else:
                    x_names.append(self.get_proxy(ascend_op.Sub, args))
            else:
                if arg is not None:
                    x_names.append(gather)
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
        return self.get_proxy(ascend_op.ConcatD, (x_names, 0))

    def get_shape_proxy(self, shape):
        if isinstance(shape, torch.fx.proxy.Proxy) or isinstance(shape, FakeTensor):
            return shape
        elif isinstance(shape, list) and symint_in_shape(shape):
            return self.process_dynamic_shape(shape)
        else:
            return self.get_proxy(
                ascend_op.Const, (shape, torch.int32, [len(shape)]))

    def get_param_proxy(self, param, type, target_shape):
        if not isinstance(param, torch.fx.proxy.Proxy) and not isinstance(param, FakeTensor):
            param = param if isinstance(param, list) else [param]
            param = self.get_proxy(
                ascend_op.Const, (param, type, [len(param)]))
        return param

    def mul_scalar(self, x, y):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        # Muls support bfloat16, int32, int16, float16, float32, complex32, complex64.
        if out_dtype not in [torch.float, torch.float16, torch.int32]:
            y_shape = list(x.node.meta['val'].shape)
            y_op = self.get_param_proxy(y, out_dtype, y_shape)
            return self.get_proxy(ascend_op.Mul, (x, y_op))
        return self.get_proxy(ascend_op.Muls, (x, y))

    def mul_complex64(self, x, y):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        assert out_dtype == torch.complex64
        # (a + bj)*(c + dj) = (ac - bd)+(ad + bc)j
        a = self.get_proxy(ascend_op.Identity, (x, 0))
        b = self.get_proxy(ascend_op.Identity, (x, 1))
        c = self.get_proxy(ascend_op.Identity, (y, 0))
        d = self.get_proxy(ascend_op.Identity, (y, 1))

        ac = self.get_proxy(ascend_op.Mul, (a, c))
        bd = self.get_proxy(ascend_op.Mul, (b, d))
        ad = self.get_proxy(ascend_op.Mul, (a, d))
        bc = self.get_proxy(ascend_op.Mul, (b, c))

        ac_bd = self.get_proxy(ascend_op.Sub, (ac, bd))
        ad_bc = self.get_proxy(ascend_op.Add, (ad, bc))

        out = self.get_proxy(ascend_op.IdentityN, (ac_bd, ad_bc))
        return out

    def binary_cmp_cast_input(self, x, y):
        if not isinstance(y, torch.fx.proxy.Proxy):
            x_dtype = x.node.meta["val"].dtype
            const_dtype = torch.float32 if x_dtype == torch.float16 else x_dtype
            y_shape = list(x.node.meta["val"].shape)
            y = self.get_param_proxy(y, const_dtype, y_shape)
            if x_dtype == torch.float16:
                y = self.get_proxy(ascend_op.Cast, (y, "FLOAT16"))
        return x, y

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
            x = self.get_param_proxy(x, None, y_shape)
        elif np.prod(x_shape) > np.prod(y_shape):
            y = self.get_param_proxy(y, None, x_shape)
        if x_dtype != out_dtype:
            x = self.get_proxy(
                ascend_op.Cast, (x, get_ascend_dtype(out_dtype)), {})
        if y_dtype != out_dtype:
            y = self.get_proxy(
                ascend_op.Cast, (y, get_ascend_dtype(out_dtype)), {})
        return self.get_proxy(ascend_op.Mul, (x, y), {})

    @register_conversion(torch.ops.aten.add.Tensor)
    def add(self, x, y, alpha: Optional[Number] = 1):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        if not isinstance(y, torch.fx.proxy.Proxy):
            y = y * alpha
            if out_dtype == torch.float or out_dtype == torch.float16:
                return self.get_proxy(ascend_op.Adds, (x, float(y)), {})
            else:
                y = self.get_proxy(ascend_op.Const, ([y], out_dtype, []))
        else:
            x_dtype = x.node.meta['val'].dtype
            y_dtype = y.node.meta['val'].dtype
            y = self.mul(y, alpha)
            if x_dtype != out_dtype:
                x = self.get_proxy(
                    ascend_op.Cast, (x, get_ascend_dtype(out_dtype)), {})
            if y_dtype != out_dtype:
                y = self.get_proxy(
                    ascend_op.Cast, (y, get_ascend_dtype(out_dtype)), {})
        return self.get_proxy(ascend_op.AddV2, (x, y), {})

    @register_conversion(torch.ops.aten.add.Scalar)
    def add_scalar(self, x, y):
        return self.add(x, y)

    @register_conversion(torch.ops.aten._to_copy.default)
    def _to_copy(self, x, dtype=None, layout=torch.strided, device=None):
        if dtype:
            if device == torch.device(type='cpu'):
                return self.get_proxy(ascend_op.CastToCpu, (x, get_ascend_dtype(dtype)))
            else:
                return self.get_proxy(ascend_op.Cast, (x, get_ascend_dtype(dtype)))
        else:
            return self.get_proxy(ascend_op.Identity, (x, None))

    @register_conversion(aten.le)
    def le(self, a, b):
        a, b = self.binary_cmp_cast_input(a, b)
        return self.get_proxy(ascend_op.LessEqual, (a, b), {})

    @register_conversion(aten.view_as_real)
    def view_as_real(self, x):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        assert out_dtype == torch.float32
        x_shape = list(x.node.meta['val'].shape)
        dim = len(x_shape)
        op1 = self.get_proxy(ascend_op.Identity, (x, 0))
        op2 = self.get_proxy(ascend_op.Identity, (x, 1))
        pack = self.get_proxy(ascend_op.Pack, ([op1, op2], dim))
        return self.get_proxy(ascend_op.Squeeze, (pack, [-1]))

    @register_conversion(aten.sqrt)
    def sqrt(self, x):
        sqrt_op = self.get_proxy(ascend_op.Sqrt, (x,))
        zero_op = self.get_proxy(ascend_op.ZerosLike, (x,))
        nan_op = self.get_proxy(ascend_op.Div, (zero_op, zero_op))
        cond_op = self.get_proxy(ascend_op.Less, (x, zero_op))
        return self.get_proxy(ascend_op.Select, (cond_op, nan_op, sqrt_op))

    @register_conversion(aten.rsqrt)
    def rsqrt(self, x):
        rsqrt_op = self.get_proxy(ascend_op.Rsqrt, (x,))
        zero_op = self.get_proxy(ascend_op.ZerosLike, (x,))
        nan_op = self.get_proxy(ascend_op.Div, (zero_op, zero_op))
        cond_op = self.get_proxy(ascend_op.Less, (x, zero_op))
        return self.get_proxy(ascend_op.Select, (cond_op, nan_op, rsqrt_op))

    @register_conversion(_operator.ge)
    def inge(self, x, y):
        if not isinstance(y, torch.fx.proxy.Proxy):
            assert isinstance(y, int)
            y = self.get_proxy(ascend_op.Const, ([y], torch.int32, []))
        return self.get_proxy(ascend_op.GreaterEqual, (x, y))

    @register_conversion(aten.div)
    def div(self, x, y):
        if isinstance(y, torch.fx.proxy.Proxy):
            return self.get_proxy(ascend_op.DivNoNan, (x, y))
        assert y != 0
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        const_dtype = torch.float32 if out_dtype == torch.float16 else out_dtype
        y_shape = list(x.node.meta['val'].shape)
        y_op = self.get_param_proxy(y, const_dtype, y_shape)
        if out_dtype == torch.float16:
            y_op = self.get_proxy(ascend_op.Cast, (y_op, "FLOAT16"), {})
        return self.get_proxy(ascend_op.Div, (x, y_op), {})

    @register_conversion(aten.slice.Tensor)
    def slice(self, x, dim=0, start=None, end=None, step=1):
        # TODO(tangzhiyi): miss step parameter
        x_shape = list(x.node.meta['val'].shape)
        y_shape = list(fx_traceback.get_current_meta()['val'].shape)
        dim = int(dim)
        start = int(start) if start is not None else 0
        start = start if start >= 0 else x_shape[dim] + start
        assert dim == -1 or dim >= 0 and dim < len(x_shape)
        assert start is None or start >= 0 and start < x_shape[dim]
        offset = [0] * len(x_shape)
        offset[dim] = start
        offset = self.get_shape_proxy(offset)
        size = self.get_shape_proxy(y_shape)
        return self.get_proxy(ascend_op.Slice, (x, offset, size))

    @register_conversion(aten.bernoulli.p)
    def Bernoulli(self, x, p, generator=None):
        assert generator is None
        dtype = x.node.meta['val'].dtype
        shape_op = self.get_proxy(ascend_op.Shape, (x,))
        prop_op = self.get_proxy(
            ascend_op.Const, ([float(p)], torch.float32, []))
        seed_op = self.get_proxy(ascend_op.Const, ([-1], torch.int64, []))
        offset_op = self.get_proxy(ascend_op.Const, ([0], torch.int64, []))
        return self.get_proxy(ascend_op.StatelessBernoulli, (shape_op, prop_op, seed_op, offset_op, dtype))

    @register_conversion(aten.new_empty_strided.default)
    def NewEmptyStrided(self, x, size, stride, dtype=torch.float32, layout=torch.strided,
                        device='cpu', pin_memory=False):
        return self.empty_like(x)

    @register_conversion(aten.empty)
    def empty(self, size, dtype=torch.int64, layout=torch.strided, device='cpu', memory_format=torch.contiguous_format):
        shape_op = self.get_proxy(
            ascend_op.Const, (size, torch.int32, [len(size)]))
        return self.get_proxy(ascend_op.Empty, (shape_op, dtype, layout, device, memory_format))

    @register_conversion(aten.empty_like.default)
    def empty_like(self, x, dtype=torch.float32, layout=torch.strided,
                   device='cpu', pin_memory=False, memory_format=torch.preserve_format):
        dtype = x.node.meta['val'].dtype
        shape = list(x.node.meta['val'].shape)
        shape_op = self.get_proxy(
            ascend_op.Const, (shape, torch.int32, [len(shape)]))
        new_memory_format=x.node.meta['tensor_meta'].memory_format if memory_format is torch.preserve_format else memory_format
        return self.get_proxy(ascend_op.Empty, (shape_op, dtype, layout, device, new_memory_format))

    @register_conversion(aten.select.int)
    def select(self, x, dim, index):
        x_shape = list(x.node.meta['val'].shape)
        y_shape = list(fx_traceback.get_current_meta()['val'].shape)
        dim = int(dim)
        index = int(index)
        assert dim >= 0 and dim < len(x_shape)
        start = index if index >= 0 else index + x_shape[dim]
        end = start + 1
        offset = [0] * len(x_shape)
        offset[dim] = start
        size = []
        for i, v in enumerate(x_shape):
            if i != dim:
                size.append(v - offset[i])
            else:
                size.append(end - offset[i])
        offset = self.get_shape_proxy(offset)
        size = self.get_shape_proxy(size)
        slice = self.get_proxy(ascend_op.Slice, (x, offset, size))
        y_shape = self.get_shape_proxy(y_shape)
        Reshape_kw = {
            "ori_op": "Select",
            "params_passed": {
                "sel_dim": dim,
            },
        }
        return self.get_proxy(ascend_op.Reshape, (slice, y_shape), Reshape_kw)

    @register_conversion(_operator.add)
    def inadd(self, x, y):
        if not isinstance(x, torch.fx.proxy.Proxy):
            assert isinstance(x, int)
            x = self.get_proxy(ascend_op.Const, (x, torch.int32))
        if not isinstance(y, torch.fx.proxy.Proxy):
            assert isinstance(y, int)
            y = self.get_proxy(ascend_op.Const, (y, torch.int32))
        return self.get_proxy(ascend_op.AddV2, (x, y))

    @register_conversion([aten.view.default, aten._unsafe_view, aten._unsafe_view.default])
    def view(self, x, size):
        result_val = fx_traceback.get_current_meta()['val']
        shape = list(result_val.shape)
        if x.node.meta["val"].dtype == torch.complex64:
            shape.append(1)
        numel = result_val.numel()
        neg = False
        for i in shape:
            if not isinstance(i, torch.SymInt):
                if i == -1:
                    neg = True
                    break
        if neg:
            prod = 1
            for i in shape:
                if not isinstance(i, torch.SymInt):
                    if i > 0:
                        prod *= i
                else:
                    raise RuntimeError(
                        "cannot handle with both negative and symint!")

            real_shape = []
            for i in shape:
                if not isinstance(i, torch.SymInt):
                    if i > 0:
                        real_shape.append(str(i))
                    else:
                        real_shape.append(str(numel / prod))
                else:
                    raise RuntimeError(
                        "cannot handle with both negative and symint!")
            shape = real_shape
        shape = self.get_shape_proxy(shape)
        if x.node.meta["val"].dtype == torch.complex64:
            real = self.get_proxy(ascend_op.Identity, (x, 0))
            imag = self.get_proxy(ascend_op.Identity, (x, 1))
            real_reshape = self.get_proxy(ascend_op.Reshape, (real, shape))
            imag_reshape = self.get_proxy(ascend_op.Reshape, (imag, shape))
            return self.get_proxy(ascend_op.IdentityN, (real_reshape, imag_reshape))
        else:
            return self.get_proxy(ascend_op.Reshape, (x, shape))
               
    @register_conversion(torch.ops.aten.where)
    def where(self, condition, x1, x2):
        # TODO(tangzhiyi): need to process scalars
        assert isinstance(x1, torch.fx.proxy.Proxy)
        assert isinstance(x2, torch.fx.proxy.Proxy)
        shape_op = self.get_proxy(ascend_op.Shape, (condition,))
        x1_bcast = self.get_proxy(ascend_op.BroadcastTo, (x1, shape_op))
        x2_bcast = self.get_proxy(ascend_op.BroadcastTo, (x2, shape_op))
        return self.get_proxy(ascend_op.Select, (condition, x1_bcast, x2_bcast))

    @register_conversion(aten.arange.default)
    def arange(self, end, start=0, step=1, dtype=None, device='xpu', layout=None, pin_memory=False):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        assert isinstance(start, torch.fx.proxy.Proxy) or type(start) in [int, float]
        assert isinstance(end, torch.fx.proxy.Proxy) or type(end) in [int, float]
        assert isinstance(step, torch.fx.proxy.Proxy) or type(step) in [int, float]

        if not isinstance(start, torch.fx.proxy.Proxy): # scalar const
            start = self.get_proxy(ascend_op.Const, (start, out_dtype))
        elif start.node.meta['val'] != out_dtype: # align tensor dtype
            start = self.get_proxy(ascend_op.Cast, (start, get_ascend_dtype(out_dtype)), {})
        if not isinstance(end, torch.fx.proxy.Proxy):
            end = self.get_proxy(ascend_op.Const, (end, out_dtype))
        elif end.node.meta['val'] != out_dtype:
            end = self.get_proxy(ascend_op.Cast, (end, get_ascend_dtype(out_dtype)), {})
        if not isinstance(step, torch.fx.proxy.Proxy):
            step = self.get_proxy(ascend_op.Const, (step, out_dtype))
        elif step.node.meta['val'] != out_dtype:
            step = self.get_proxy(ascend_op.Cast, (step, get_ascend_dtype(out_dtype)), {})
        return self.get_proxy(ascend_op.Range, (start, end, step))

    @register_conversion(aten.arange.start)
    def arange_start(self, start, end, step=1, dtype=None, device=None, layout=None, pin_memory=False):
        return self.arange(end, start)

    @register_conversion([aten.eq, aten.eq.Tensor])
    def eq(self, a, b):
        a, b = self.binary_cmp_cast_input(a, b)
        return self.get_proxy(ascend_op.Equal, (a, b))

    @register_conversion(aten.ne.Scalar)
    def ne(self, a, b):
        a, b = self.binary_cmp_cast_input(a, b)
        return self.get_proxy(ascend_op.NotEqual, (a, b))

    @register_conversion([aten.lt.Scalar, aten.lt.Tensor])
    def lt(self, x, y):
        x, y = self.binary_cmp_cast_input(x, y)
        return self.get_proxy(ascend_op.Less, (x, y))

    @register_conversion(aten.masked_fill.Scalar)
    def masked_fill(self, x, mask, value):
        x_dtype = x.node.meta['val'].dtype
        const_dtype = torch.float32 if x_dtype == torch.float16 else x_dtype
        if str(value) == "-inf":
            value = -3.4028234663852886e+38
        mask_shape = list(mask.node.meta['val'].shape)
        value = self.get_param_proxy(value, const_dtype, mask_shape)
        if x_dtype == torch.float16:
            value = self.get_proxy(ascend_op.Cast, (value, "FLOAT16"))
        return self.get_proxy(ascend_op.MaskedFill, (x, mask, value))

    @register_conversion([torch.ops.aten.scatter.src, torch.ops.aten.scatter.value])
    def scatter(self, var, dim, index, value):
        assert isinstance(dim, int)
        index_shape = list(index.node.meta['val'].shape)
        if isinstance(value, torch.fx.proxy.Proxy):
            preprocess = self.get_shape_proxy(index_shape)
            value = self.get_proxy(ascend_op.Reshape, (value, preprocess))
        else:
            out_dtype = fx_traceback.get_current_meta()['val'].dtype
            value = self.get_proxy(ascend_op.Const, (value, out_dtype))
            shape = self.get_proxy(ascend_op.Shape, (index,))
            value = self.get_proxy(ascend_op.BroadcastTo, (value, shape))
        return self.get_proxy(ascend_op.ScatterElements, (var, index, value, dim))

    @register_conversion(torch.ops.aten.nll_loss_forward.default)
    def nll_loss_forward(self, x, target, weight, reduction, ignore_index):
        assert weight is None
        assert ignore_index == -100
        reduction_str = get_reduction_str(reduction)
        csize = [list(x.node.meta['val'].shape)[1]]
        target = self.get_proxy(ascend_op.Cast, (target, "INT32"))
        weight = self.get_proxy(ascend_op.FillV2D, (1.0, csize))
        return self.get_proxy(ascend_op.NLLLoss, (x, target, weight, reduction_str, ignore_index))

    @register_conversion(torch.ops.aten.nll_loss_backward.default)
    def nll_loss_backward(self, grad_output, x, target, weight, reduction, ignore_index, total_weight):
        assert weight is None
        assert ignore_index == -100
        reduction_str = get_reduction_str(reduction)
        csize = [list(x.node.meta['val'].shape)[1]]
        target = self.get_proxy(ascend_op.Cast, (target, "INT32"))
        weight = self.get_proxy(ascend_op.FillV2D, (1.0, csize))
        return self.get_proxy(ascend_op.NLLLossGrad, (x, grad_output, target,
                                                      weight, total_weight,
                                                      reduction_str, ignore_index))

    @register_conversion(torch.ops.aten.cat.default)
    def cat(self, x, dim=0):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        x_list = []
        for i, v in enumerate(x):
            dtype = v.node.meta['val'].dtype
            if dtype == out_dtype:
                x_list.append(x[i])
                continue
            # cast to y_dtype
            x_list.append(self.get_proxy(ascend_op.Cast,
                          (x[i], get_ascend_dtype(out_dtype))))
        return self.get_proxy(ascend_op.ConcatD, (x_list, dim))

    @register_conversion(torch.ops.aten.threshold_backward.default)
    def threshold_backward(self, grad_output, x, threshold):
        if threshold == 0:
            return self.get_proxy(ascend_op.ReluGrad, (grad_output, x))
        else:
            return self.get_proxy(ascend_op.ThresholdGradV2D, (grad_output, x, threshold))

    @register_conversion(aten.view_as_complex.default)
    def view_as_complex(self, x):
        x_val = x.node.meta['val']
        x_shape = list(x_val.shape)
        assert x_val.dtype == torch.float32
        assert x_shape[-1] == 2
        dim = len(x_shape) - 1
        splitD_kw = { "from_view_complex": True }
        return self.get_proxy(ascend_op.SplitD, (x, dim, 2, 2), splitD_kw)

    @register_conversion(torch.ops.aten.full.default)
    def full(self, dims, value, dtype=torch.float32, layout=torch.strided,
             device='cpu', pin_memory=False, memory_format=torch.preserve_format):
        if len(dims) == 0:
            dims = [1]
        torch_dtype = dtype if dtype else torch.get_default_dtype()
        dims = [dim.node.meta['val'] if isinstance(dim, torch.fx.proxy.Proxy) and hasattr(
            dim.node, 'meta') else dim for dim in dims]
        if isinstance(value, torch.fx.proxy.Proxy) and hasattr(value.node, 'meta'):
            value = value.node.meta['val']
        dims = self.get_shape_proxy(dims)
        value = self.get_proxy(ascend_op.Const, ([value], torch_dtype, []))
        return self.get_proxy(ascend_op.Fill, (dims, value))

    @register_conversion(torch.ops.aten.fill.Scalar)
    def fills(self, x, value):
        return self.get_proxy(ascend_op.Fills, (x, value))

    @register_conversion(torch.ops.aten.topk.default)
    def topk(self, x, k, dim=-1, largest=True, sorted=True):
        if not isinstance(k, torch.fx.proxy.Proxy):
            k = self.get_proxy(ascend_op.Const, ([k], torch.int32, []))
        return self.get_proxy(ascend_op.TopK, (x, k, dim, largest, sorted))

    @register_conversion(torch.ops.aten.sort.default)
    def sort(self, x, dim=-1, descending=False):
        return self.get_proxy(ascend_op.Sort, (x, dim, descending))

    @register_conversion(torch.ops.aten.ones.default)
    def ones(self, shape, dtype=torch.float32, layout=torch.strided, device='cpu', pin_memory=False):
        shape = self.get_proxy(
            ascend_op.Const, (shape, torch.int32, [len(shape)]))
        like = self.get_proxy(ascend_op.Empty, (shape, dtype, layout, device))
        return self.get_proxy(ascend_op.OnesLike, (like,))

    @register_conversion(torch.ops.aten.new_ones.default)
    def new_ones(self, x, shape, dtype=torch.int64, layout=None, device='cpu', pin_memory=False):
        return self.ones(shape, dtype)

    def index_base(self, x, dim, index):
        dim = [dim] if not isinstance(dim, list) else dim
        if isinstance(index, list):
            assert len(index) == 1
            index = index[0]
        assert dim[0] == 0
        dim_op = self.get_proxy(
            ascend_op.Const, (dim, torch.int32, [len(dim)]))
        return self.get_proxy(ascend_op.GatherV2, (x, index, dim_op))

    @register_conversion(torch.ops.aten.index.Tensor)
    def index(self, x, index):
        return self.index_base(x, 0, index)

    @register_conversion(torch.ops.aten.index_select.default)
    def index_arg3_(self, x, dim, index):
        return self.index_base(x, dim, index)

    @register_conversion(torch.ops.aten._native_batch_norm_legit_functional.default)
    def _native_batch_norm_legit_functional(self, x, weight, bias, running_mean, running_var,
                                            train, momentum, eps):
        if train is False:
            raise RuntimeError("not supported yet!")
        x_shape = list(x.node.meta['val'].shape)
        x_dtype = x.node.meta['val'].dtype
        # TODO(tangzhiyi): now assume output name is y.
        # TODO(daoxin): potential dynamic shape issue in resnet18
        bn_reduce = self.get_proxy(ascend_op.BNTrainingReduce,
                                   (x, x_shape, "NCHW", x_dtype))
        # 2. call BNTrainingUpdate
        bn_update = self.get_proxy(ascend_op.BNTrainingUpdate,
                                   (x, bn_reduce, 0,
                                    bn_reduce, 1, weight, bias,
                                    running_mean, running_var, eps, momentum))
        # 3. tie all results: result, saved_mean, saved_invstd
        edges = {bn_update.node.name: [
            "y", "batch_mean", "batch_variance", "mean", "variance"]}
        return self.get_proxy(ascend_op.IdentityN, (bn_update,), edges)

    @register_conversion(torch.ops.aten.native_batch_norm_backward.default)
    def native_batch_norm_backward(self, grad_out, x, weight, running_mean, running_var,
                                   save_mean, save_invstd, train, eps, grad_input_mask):
        x_shape = list(x.node.meta["val"].shape)
        x_dtype = x.node.meta["val"].dtype
        # get grad_weight and grad_bias
        update_grad = self.get_proxy(ascend_op.BNTrainingUpdateGrad,
                                     (grad_out, x_shape, "NCHW", x_dtype, "backprops",
                                      x, save_mean, save_invstd, eps))
        # get grad_input
        reduce_grad = self.get_proxy(ascend_op.BNTrainingReduceGrad,
                                     (grad_out, x, update_grad, 0, update_grad, 1,
                                      weight, save_mean, save_invstd, eps))
        for mask in grad_input_mask:
            assert mask is True
        edges = {reduce_grad.node.name: ["y"],
                 update_grad.node.name: ["diff_scale", "diff_offset"]}
        return self.get_proxy(ascend_op.IdentityN, (reduce_grad, update_grad), edges)

    @register_conversion(torch.ops.prims.convert_element_type)
    def convert_element_type(self, x, dtype):
        return self.get_proxy(ascend_op.Cast, (x, get_ascend_dtype(dtype)))

    @register_conversion(torch.ops.aten.convolution_backward)
    def convolution_backward(self, grad, input, weight, bias,
                             stride, padding, dilation, transposed,
                             output_padding, groups, grad_input_mask):
        assert transposed is False
        assert output_padding == [0, 0]
        stride = [1, 1, stride[0], stride[1]]
        padding = [padding[0], padding[0], padding[1], padding[1]]
        dilation = [1, 1, dilation[0], dilation[1]]
        data_format = "NCHW"
        if grad_input_mask[0] is True:
            input_shape = self.get_proxy(ascend_op.Shape, (input,))
            input_op = self.get_proxy(ascend_op.Conv2DBackpropInput,
                                      (input_shape, weight, grad, stride, padding,
                                       dilation, groups, data_format))
        if grad_input_mask[1] is True:
            filter_shape = self.get_proxy(ascend_op.Shape, (weight,))
            filter_op = self.get_proxy(ascend_op.Conv2DBackpropFilter,
                                       (input, filter_shape, grad, stride, padding,
                                        dilation, data_format))
        # TODO(tangzhiyi): bias is not supported yet
        assert grad_input_mask[2] is False
        outputs = []
        outputs.append(input_op if grad_input_mask[0] else filter_op)
        outputs.append(filter_op if grad_input_mask[1] else input_op)
        return self.get_proxy(ascend_op.IdentityN, outputs)

    @register_conversion(torch.ops.aten.max_pool2d_with_indices_backward)
    def maxpool2dbackward(self, grad, x, kernel_size, stride, padding, dilation,
                          ceil_mode, index):
        assert len(kernel_size) == 2 or len(kernel_size) == 1
        assert len(stride) == 2 or len(stride) == 1
        assert len(padding) == 2 or len(padding) == 1
        assert len(dilation) == 2 or len(dilation) == 1
        assert dilation == [1, 1]
        kernel_size = [1, 1, kernel_size[0], kernel_size[1]
                       if len(kernel_size) == 2 else kernel_size[0]]
        stride = [1, 1, stride[0], stride[1]
                  if len(stride) == 2 else stride[0]]
        padding = [1, padding[0], padding[1] if len(
            padding) == 2 else padding[0], 1]
        if padding != [0, 0]:
            padding = [0, 0, 0, 0, padding[0],
                       padding[0], padding[1], padding[1]]
            padding_const = self.get_proxy(
                ascend_op.Const, (padding, torch.int32, [8], "NCHW"))
            pad_op = self.get_proxy(ascend_op.PadV3, (x, padding_const, ))
            fwd_out = self.get_proxy(ascend_op.MaxPool,
                                     (pad_op, kernel_size, stride, "VALID", "NCHW"))
            bwd = self.get_proxy(ascend_op.MaxPoolGrad,
                                 (pad_op, fwd_out, grad, kernel_size, stride, "VALID", "NCHW"))
            return self.get_proxy(ascend_op.PadV3Grad, (bwd, padding_const))
        else:
            fwd_out = self.get_proxy(ascend_op.MaxPool,
                                     (x, kernel_size, stride, "VALID", "NCHW"))
            return self.get_proxy(ascend_op.MaxPoolGrad,
                                  (x, fwd_out, grad, kernel_size, stride, "VALID", "NCHW"))

    @register_conversion(torch.ops.aten.max_pool2d_with_indices)
    def maxpool2d(self, x, ksize, strides, padding=[0, 0]):
        assert len(ksize) == 2
        assert len(strides) == 2
        ksize = [1, 1, ksize[0], ksize[1]]
        strides = [1, 1, strides[0], strides[1]]
        if padding != [0, 0]:
            padding = [0, 0, 0, 0, padding[0],
                       padding[0], padding[1], padding[1]]
            paddings_const = self.get_proxy(
                ascend_op.Const, (padding, [4, 2], torch.int32, "NCHW"))
            x = self.get_proxy(ascend_op.PadV3, (x, paddings_const))
        fwd_out_op = self.get_proxy(
            ascend_op.MaxPool, (x, ksize, strides, "VALID", "NCHW"))
        shape_op = self.get_proxy(ascend_op.Shape, (fwd_out_op,))
        index_op = self.get_proxy(ascend_op.Empty, (shape_op, torch.int64))
        return self.get_proxy(ascend_op.IdentityN, (fwd_out_op, index_op))

    @register_conversion(aten.expand.default)
    def expand(self, x, shape):
        x_shape = list(x.node.meta['val'].shape)
        y_shape = list(fx_traceback.get_current_meta()['val'].shape)
        if x_shape == y_shape:
            return self.get_proxy(ascend_op.Identity, (x, None))
        if x.node.meta['val'].dtype == torch.int64:
            x = self.get_proxy(ascend_op.Cast, (x, "INT32"))
        shape = [dim.node.meta['val'] if hasattr(
            dim, 'node') else dim for dim in shape]
        if isinstance(shape, list) and symint_in_shape(shape):
            preprocess_shape = self.process_dynamic_shape(shape)
            return self.get_proxy(ascend_op.Expand, (x, preprocess_shape))
        else:
            return self.get_proxy(ascend_op.ExpandD, (x, shape))

    @register_conversion(torch.ops.aten.slice_backward.default)
    def slice_backward(self, grad, input_shape, dim, start, end, step):
        start = start if start >= 0 else input_shape[dim] + start
        assert step == 1
        assert dim >= 0 and dim < len(input_shape)
        assert start >= 0 and start < input_shape[dim]
        rank = len(input_shape)
        end = end if end <= input_shape[dim] else input_shape[dim]
        end = end if end >= 0 else end + input_shape[dim]
        pad = np.zeros((rank, 2), dtype=np.int32)
        for i, v in enumerate(input_shape):
            if i == dim:
                pad[i][0] = start
                pad[i][1] = v - end
        padding_const = self.get_proxy(ascend_op.Const,
                                       (pad.flatten().tolist(), torch.int32, [rank, 2]))
        return self.get_proxy(ascend_op.Pad, (grad, padding_const))

    @register_conversion(torch.ops.aten.var)
    def var(self, x, axes=[], correction=1, keepdim=True):
        if correction == 1:
            unbiased = True
        elif correction == 0:
            unbiased = False
        else:
            raise RuntimeError("not supported yet!")
        if not isinstance(axes, list):
            axes = [axes]
        axes_op = self.get_proxy(ascend_op.Const, (axes, torch.int32, [
                                 len(axes)] if len(axes) > 0 else []))
        mean_op = self.get_proxy(ascend_op.ReduceMean, (x, axes_op))
        input_shape_op = self.get_proxy(ascend_op.Shape, (x,))
        broadcast_op = self.get_proxy(
            ascend_op.BroadcastTo, (mean_op, input_shape_op))
        return self.get_proxy(ascend_op.ReduceStdV2Update, (x, broadcast_op, axes, unbiased, keepdim))

    @register_conversion(torch.ops.aten.pow)
    def pow(self, x, exp):
        if isinstance(exp, torch.fx.proxy.Proxy):
            return self.get_proxy(ascend_op.Pow, (x, exp))
        # exp is scalar
        dtype = fx_traceback.get_current_meta()['val'].dtype
        exp_const = self.get_proxy(ascend_op.Const, ([exp], dtype, []))
        return self.get_proxy(ascend_op.Pow, (x, exp_const))

    @register_conversion(aten.maximum.default)
    def maximum(self, a, b):
        a_shape = list(a.node.meta['val'].shape)
        b_shape = list(b.node.meta['val'].shape)
        if np.prod(b_shape) < np.prod(a_shape):
            b = self.get_param_proxy(b, None, a_shape)
        if a.node.meta['val'].dtype == torch.float16:
            b = self.get_proxy(ascend_op.Cast, (b, "FLOAT16"))
        return self.get_proxy(ascend_op.Maximum, (a, b))

    def common_process_scalar(self, y, dtype):
        need_cast = False
        if dtype == torch.float16:
            dtype = torch.float32
            need_cast = True
        y = self.get_proxy(ascend_op.Const, (y, dtype))
        if need_cast:
            y = self.get_proxy(ascend_op.Cast, (y, "FLOAT16"))
        return y

    @register_conversion(aten.sub)
    def sub(self, x, y):
        if not isinstance(y, torch.fx.proxy.Proxy):
            y = self.common_process_scalar(y, x.node.meta['val'].dtype)
        return self.get_proxy(ascend_op.Sub, (x, y))

    @register_conversion(aten.rsub)
    def rsub(self, x, y):
        if not isinstance(y, torch.fx.proxy.Proxy):
            y = self.common_process_scalar(y, x.node.meta['val'].dtype)
        return self.get_proxy(ascend_op.Sub, (y, x))

    @register_conversion(aten.transpose.int)
    def transpose(self, input, dim0, dim1):
        input_shape = list(input.node.meta['val'].shape)
        rank = len(input_shape)
        dim0 = int(dim0)
        dim1 = int(dim1)
        perm = [num for num in range(rank)]
        perm[dim0] = dim1
        perm[dim1] = dim0
        perm = self.get_shape_proxy(perm)
        return self.get_proxy(ascend_op.Transpose, (input, perm))

    @register_conversion(torch.ops.aten.convolution)
    def convolution(self, input, weight, bias, stride, padding,
                    dilation, transposed, output_padding, groups):
        assert transposed is False
        assert output_padding == [0, 0]

        if len(stride) == 2:
            stride = [1, 1, stride[0], stride[1]]
        if len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]
        if len(dilation) == 2:
            dilation = [dilation[0], dilation[0], dilation[1], dilation[1]]
        out_node = fx_traceback.get_current_meta()
        format = "NCHW" if out_node['val'].stride()[-1] == 1 else "NHWC"
        return self.get_proxy(ascend_op.Conv2D, (input, weight, stride, padding,
                                                 dilation, groups, format, bias))

    @register_conversion(_operator.mul)
    def inmul(self, x, y):
        assert (not isinstance(y, torch.fx.proxy.Proxy))
        y = self.get_proxy(ascend_op.Const, ([y], torch.int32, []))
        return self.get_proxy(ascend_op.Mul, (x, y))

    @register_conversion(torch.ops.aten.sym_size)
    def symsize(self, x, dim):
        dim = [dim] if not isinstance(dim, list) else dim
        shape = self.get_proxy(ascend_op.Shape, (x,))
        axis = self.get_proxy(ascend_op.Const, ([0], torch.int32, [1]))
        indices = self.get_proxy(
            ascend_op.Const, (dim, torch.int32, [len(dim)]))
        return self.get_proxy(ascend_op.GatherV2, (shape, indices, axis))

    @register_conversion(torch.ops.aten.mm.default)
    def mm(self, x, y):
        # TODO! MatMul not support fp32 input
        # for higher precision in some cases
        if len(self.sym_in_args) > 0 or len(self.sym_to_inputs) > 0:
            x = self.get_proxy(ascend_op.Unsqueeze, (x, [0]))
            y = self.get_proxy(ascend_op.Unsqueeze, (y, [0]))
            mm = self.get_proxy(ascend_op.BatchMatMul, (x, y, False, False))
            return self.get_proxy(ascend_op.Squeeze, (mm, [0]))
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        trans_x = False
        trans_y = False
        if isinstance(x.node.target, ascend_op.Permute) and x.node.args[1] == [1, 0]:
            x = self.get_proxy_from_node(x.node.args[0])
            trans_x = True
        if isinstance(y.node.target, ascend_op.Permute) and y.node.args[1] == [1, 0]:
            y = self.get_proxy_from_node(y.node.args[0])
            trans_y = True
        mm = self.get_proxy(ascend_op.MatMul, (x, y, trans_x, trans_y))
        return self.get_proxy(ascend_op.Cast, (mm, get_ascend_dtype(out_dtype)))

    @register_conversion(aten.bmm.default)
    def bmm(self, x, y):
        out_dtype = fx_traceback.get_current_meta()['val'].dtype
        bmm = self.get_proxy(ascend_op.BatchMatMul, (x, y, False, False))
        return self.get_proxy(ascend_op.Cast, (bmm, get_ascend_dtype(out_dtype)))

    @register_conversion(torch.torch.ops.aten.addmm)
    def addmm(self, c, a, b, beta=1.0, alpha=1.0):
        beta_op = self.get_proxy(ascend_op.Const, ([beta], torch.float32, []))
        alpha_op = self.get_proxy(
            ascend_op.Const, ([alpha], torch.float32, []))
        c_beta_op = self.get_proxy(ascend_op.Mul, (c, beta_op))
        a_alpha_op = self.get_proxy(ascend_op.Mul, (a, alpha_op))
        matmul_op = self.get_proxy(
            ascend_op.MatMul, (a_alpha_op, b, False, False))
        return self.get_proxy(ascend_op.Add, (c_beta_op, matmul_op))

    @register_conversion(torch.ops.aten.mean)
    def mean(self, x, dims=[], keepdim=False):
        if not isinstance(dims, list):
            dims = [dims]
        return self.get_proxy(ascend_op.ReduceMeanD, (x, dims, keepdim, False))

    @register_conversion(torch.ops.aten.cumsum.default)
    def cumsum(self, x, dim, dtype=None):
        dim_const = self.get_proxy(ascend_op.Const, ([dim], torch.int32, [1]))
        return self.get_proxy(ascend_op.Cumsum, (x, dim_const))

    @register_conversion(torch.ops.aten._log_softmax.default)
    def log_softmax(self, x, dim, half_to_float):
        assert half_to_float is False
        dim = [dim] if not isinstance(dim, list) else dim
        return self.get_proxy(ascend_op.LogSoftmaxV2, (x, dim))

    @register_conversion(torch.ops.aten._log_softmax_backward_data.default)
    def log_softmax_backward_data(self, grad_output, output, dim, input_dtype):
        dim = [dim] if not isinstance(dim, list) else dim
        return self.get_proxy(ascend_op.LogSoftmaxGrad, (grad_output, output, dim))

    @register_conversion(torch.ops.aten.repeat_interleave)
    def repeat_interleave(self, repeats, output_size=1):
        x_shape = list(repeats.node.meta['val'].shape)
        assert len(x_shape) == 1
        assert x_shape[0] == 1
        # TODO! fix implementation of repeatinterleave
        # Consider situation for repeats > 1
        return self.get_proxy(ascend_op.Const, ([0], torch.int64, [1]))

    @register_conversion([aten.lift_fresh_copy, aten.lift_fresh_copy.default])
    def lift_fresh_copy(self, tensor_constant):
        return self.get_proxy(ascend_op.Identity, (tensor_constant, None))

    @register_conversion(torch.ops.aten.clone)
    def clone(self, a, memory_format=torch.contiguous_format):
        return self.get_proxy(ascend_op.Identity, (a, None))

    @register_conversion(torch.ops.aten.copy_)
    def copy_(self, dst, src):
        return self.get_proxy(ascend_op.IdentityInp, (src, dst))

    @register_conversion(torch.ops.aten.copy)
    def copy(self, dst, src):
        return self.get_proxy(ascend_op.Identity, (src, None))

    @register_conversion(torch.ops.aten.unsqueeze)
    def unsqueeze(self, x, dim):
        if not isinstance(dim, list):
            dim = [dim]
        return self.get_proxy(ascend_op.Unsqueeze, (x, dim))

    @register_conversion(torch.ops.aten.squeeze)
    def squeeze(self, x, dim):
        if not isinstance(dim, list):
            dim = [dim]
        return self.get_proxy(ascend_op.Squeeze, (x, dim))

    @register_conversion(torch.ops.aten.exp)
    def exp(self, a):
        return self.get_proxy(ascend_op.Exp, (a,))

    @register_conversion(torch.ops.aten.embedding.default)
    def embedding(self, weight, indices, padding_idx=-1):
        # TODO! consider situation for padding_idx
        # during training stage
        axis = self.get_proxy(ascend_op.Const, ([0], torch.int32, [1]))
        return self.get_proxy(ascend_op.GatherV2, (weight, indices, axis))

    @register_conversion(torch.ops.aten.gather)
    def gather(self, x, dim, index):
        return self.get_proxy(ascend_op.GatherElements, (x, index, dim))

    @register_conversion(aten.t.default)
    def t(self, input):
        shape = fx_traceback.get_current_meta()['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape.reverse()
        return self.get_proxy(ascend_op.Permute, (input, permute_shape))

    @register_conversion(torch.ops.aten.permute)
    def permute(self, x, dims):
        if dims is not None and not isinstance(dims, list):
            dims = [dims]
        return self.get_proxy(ascend_op.Permute, (x, dims))

    @register_conversion(torch.ops.aten._softmax)
    def _softmax(self, x, dim=-1, half_to_float=False):
        if isinstance(dim, int):
            dim = [dim]
        assert (half_to_float is False)
        return self.get_proxy(ascend_op.SoftmaxV2, (x, dim))

    @register_conversion(torch.ops.aten.sum.default)
    def sum(self, a):
        return self.sumdim(a)

    @register_conversion(torch.ops.aten.sum.dim_IntList)
    def sumdim(self, x, dims=[], keepdim=False, dtype=None):
        x_dtype = x.node.meta['val'].dtype
        if not isinstance(dims, list):
            dims = [dims]
        if dtype is None or x_dtype == dtype:
            return self.get_proxy(ascend_op.ReduceSumD, (x, dims, keepdim))
        sum = self.get_proxy(ascend_op.ReduceSumD, (x, dims, keepdim))
        return self.get_proxy(ascend_op.Cast, (sum, get_ascend_dtype(dtype)))

    @register_conversion(torch.ops.aten.amax)
    def amax(self, x, dims, keepdim=False):
        if not isinstance(dims, list):
            dims = [dims]
        return self.get_proxy(ascend_op.ReduceMaxD, (x, dims, keepdim))

    @register_conversion(torch.ops.aten._softmax_backward_data.default)
    def softmax_backward_data(self, grad_output, output, dim, input_dtype):
        dim = [dim] if not isinstance(dim, list) else dim
        return self.get_proxy(ascend_op.SoftmaxGrad, (grad_output, output, dim))

    @register_conversion(aten.log)
    def log(self, x):
        return self.get_proxy(ascend_op.Log, (x,))

    @register_conversion(aten.stack)
    def stack(self, x, dim):
        return self.get_proxy(ascend_op.Pack, (x, dim))

    @register_conversion(torch.ops.aten.neg)
    def neg(self, a):
        return self.get_proxy(ascend_op.Neg, (a,))

    @register_conversion(torch.ops.aten.relu)
    def relu(self, a):
        return self.get_proxy(ascend_op.Relu, (a,))

    @register_conversion(torch.ops.aten.silu)
    def silu(self, a):
        return self.get_proxy(ascend_op.Swish, (a, 1.0))

    @register_conversion(torch.ops.aten.sigmoid)
    def sigmoid(self, x):
        return self.get_proxy(ascend_op.Sigmoid, (x,))

    @register_conversion(operator.getitem)
    def identity(self, x, idx):
        return self.get_proxy(ascend_op.Identity, (x, idx))

    @register_conversion(torch.ops.aten.full_like)
    def fulllike(self, x, value, dtype=torch.float32, layout=torch.strided,
                 device='cpu', pin_memory=False, memory_format=torch.preserve_format):
        return self.get_proxy(ascend_op.Fills, (x,float(value)))

    @register_conversion(torch.ops.aten.zeros_like.default)
    def zeros_like(self, x, dtype=torch.float32, layout=torch.strided,
                   device='cpu', pin_memory=False, memory_format=torch.preserve_format):
        return self.get_proxy(ascend_op.ZerosLike, (x,))

    @register_conversion(torch.ops.aten.rand_like.default)
    def RandLike(self, x, dtype=torch.float32, layout=torch.strided,
                 device='cpu', pin_memory=False, memory_format=torch.preserve_format):
        ascend_dtype = get_ascend_dtype(x.node.meta['val'].dtype)
        key_op = self.get_proxy(ascend_op.Const, ([0], torch.int32, [1]))
        key_cast_op = self.get_proxy(ascend_op.Cast, (key_op, "UINT64"))
        counter_op = self.get_proxy(
            ascend_op.Const, ([0, 0], torch.int32, [2]))
        counter_cast_op = self.get_proxy(
            ascend_op.Cast, (counter_op, "UINT64"))
        alg_op = self.get_proxy(ascend_op.Const, ([1], torch.int32, []))
        shape_op = self.get_proxy(ascend_op.Shape, (x,))
        return self.get_proxy(ascend_op.StatelessRandomUniformV2, (shape_op, key_cast_op,
                                                                   counter_cast_op, alg_op,
                                                                   ascend_dtype))

    @register_conversion(torch.ops.aten.gt.Scalar)
    def GtScalar(self, x, y):
        dtype = get_ascend_dtype(x.node.meta['val'].dtype)
        scalar_op = self.get_proxy(
            ascend_op.Const, ([float(y)], torch.float, []))
        cast_op = self.get_proxy(ascend_op.Cast, (scalar_op, dtype))
        return self.get_proxy(ascend_op.Greater, (x, cast_op))

    @register_conversion(torch.ops.aten.addcmul.default)
    def AddCMul(self, a, b, c, value=1):
        dtype = a.node.meta['val'].dtype
        not_support_type = False
        orig_ascend_dtype = get_ascend_dtype(dtype)
        try:
            cpp_dtype = get_cpp_dtype(dtype)
        except Exception:
            not_support_type = True
        value_op = None
        if not_support_type:
            const_op = self.get_proxy(
                ascend_op.Const, ([float(value)], torch.float32, []))
            value_op = self.get_proxy(
                ascend_op.Cast, (const_op, orig_ascend_dtype))
        else:
            value_op = self.get_proxy(ascend_op.Const, ([value], dtype, []))
        return self.get_proxy(ascend_op.Addcmul, (a, b, c, value_op))

    @register_conversion(torch.ops.aten.reciprocal.default)
    def Reciprocal(self, x):
        return self.get_proxy(ascend_op.Reciprocal, (x,))

    @register_conversion(torch.ops.aten.native_dropout.default)
    def NativeDropout(self, x, p, train):
        assert train is True
        dtype = x.node.meta['val'].dtype
        p = 1. - p
        shape = self.get_proxy(ascend_op.Shape, (x,))
        prob = self.get_proxy(ascend_op.Const, ([float(p)], torch.float, []))
        mask = self.get_proxy(ascend_op.DropOutGenMaskV4, (shape, prob))
        prob_op = prob
        if dtype == torch.float16:
            cast = self.get_proxy(ascend_op.Cast, (prob, "FLOAT16"))
            prob_op = cast
        do_mask = self.get_proxy(ascend_op.DropOutDoMaskV3, (x, mask, prob_op))
        return self.get_proxy(ascend_op.IdentityN, (do_mask, mask))

    @register_conversion(torch.ops.aten.native_dropout_backward.default)
    def NativeDropoutBackward(self, grad_output, mask, scale):
        dtype = grad_output.node.meta['val'].dtype
        p = 1. - scale
        prob = self.get_proxy(ascend_op.Const, ([float(p)], torch.float, []))
        prob_op = prob
        if dtype == torch.float16:
            cast = self.get_proxy(ascend_op.Cast, (prob, "FLOAT16"))
            prob_op = cast
        return self.get_proxy(ascend_op.DropOutDoMaskV3, (grad_output, mask, prob_op))
