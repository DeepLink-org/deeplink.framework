import torch
import numpy as np
import _operator
from typing import Tuple
from contextlib import nullcontext
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._functorch import config
from torch.utils._pytree import tree_map, tree_flatten
from abc import ABC, abstractmethod
from dicp.dynamo_bridge.utils import TensorInfo, get_memory_format
from dicp.dynamo_bridge.operator import Operator


aten = torch.ops.aten


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def negative_in_shape(shape):
    for elem in shape:
        if elem < 0:
            return True
    return False


class Adds(Operator):
    def __init__(self):
        super().__init__("adds")
        # self.torch_op = aten.add


class Add(Operator):
    def __init__(self):
        super().__init__("add")
        # self.torch_op = aten.add

    # def __call__(self, a, b):
    #     if hasattr(a, 'meta'):
    #         a = a.meta['val']
    #         a_shape = a.shape
    #     else:
    #         a_shape = [1]
    #     if hasattr(b, 'meta'):
    #         b = b.meta['val']
    #         b_shape = b.shape
    #     else:
    #         b_shape = [1]

    #     fake_mode = None
    #     for arg in [a, b]:
    #         if isinstance(arg, FakeTensor):
    #             fake_mode = arg.fake_mode
    #             break
    #     fake_mode = self.fake_mode if fake_mode is None else fake_mode

    #     # TODO! better to check
    #     # whether satisfy broadcast
    #     if np.prod(a_shape) > np.prod(b_shape):
    #         shape = a_shape
    #     else:
    #         shape = b_shape
    #     with fake_mode:
    #         return aten.empty(shape, dtype=a.dtype)


class BroadcastTo(Operator):
    def __init__(self):
        super().__init__("BroadcastTo")


class Range(Operator):
    def __init__(self):
        super().__init__("Range")


class CumSum(Operator):
    def __init__(self):
        super().__init__("Cumsum")
        self.torch_op = aten.cumsum.default


class MatMul(Operator):
    def __init__(self):
        super().__init__("MatMul")
        self.torch_op = aten.mm

    # def infer_result(self, a, b, trans_a=False, trans_b=False, change_input=False):
    #     if hasattr(a, 'meta'):
    #         a = a.meta['val']
    #     if hasattr(b, 'meta'):
    #         b = b.meta['val']
    #     if change_input:
    #         (a, b) = (b, a)

    #     trans_a_shape = shape_functions.t(a.shape) if trans_a else a.shape
    #     trans_b_shape = shape_functions.t(b.shape) if trans_b else b.shape
    #     mm_shape = shape_functions.matmul(trans_a_shape, trans_b_shape)
    #     return TensorInfo(mm_shape, dtype=a.dtype, memory_format=get_memory_format(a))


class BatchMatMul(Operator):
    def __init__(self):
        super().__init__("BatchMatMul")
        self.torch_op = aten.bmm

    # def infer_result(self, x1, x2, adj_x1=False, adj_x2=False):
    #     if hasattr(x1, 'meta'):
    #         x1 = x1.meta['val']
    #     if hasattr(x2, 'meta'):
    #         x2 = x2.meta['val']
    #     trans_x1_shape = shape_functions.transpose(x1.shape, 1, 2) if adj_x1 else x1.shape
    #     trans_x2_shape = shape_functions.transpose(x2.shape, 1, 2) if adj_x2 else x2.shape
    #     bmm_shape = shape_functions.bmm(trans_x1_shape, trans_x2_shape)
    #     return TensorInfo(bmm_shape, dtype=x1.dtype, memory_format=get_memory_format(x1))


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")
        # self.torch_op = aten.sub


class Mul(Operator):
    def __init__(self):
        super().__init__("Mul")


class Div(Operator):
    def __init__(self):
        super().__init__("Div")
        self.torch_op = aten.div


class DivNoNan(Operator):
    def __init__(self):
        super().__init__("DivNoNan")


class Maximum(Operator):
    def __init__(self):
        super().__init__("Maximum")
        self.torch_op = aten.maximum


class Rsqrt(Operator):
    def __init__(self):
        super().__init__("Rsqrt")
        self.torch_op = aten.rsqrt


class Sqrt(Operator):
    def __init__(self):
        super().__init__("Sqrt")
        self.torch_op = aten.sqrt


class Log(Operator):
    def __init__(self):
        super().__init__("Log")
        self.torch_op = aten.log


class Exp(Operator):
    def __init__(self):
        super().__init__("Exp")
        self.torch_op = aten.exp


class Neg(Operator):
    def __init__(self):
        super().__init__("Neg")
        self.torch_op = aten.neg


class Relu(Operator):
    def __init__(self):
        super().__init__("Relu")
        self.torch_op = aten.relu


class Swish(Operator):
    def __init__(self):
        super().__init__("Swish")
        self.torch_op = aten.silu


class Transpose(Operator):
    def __init__(self):
        super().__init__("Transpose")

    def infer_result(self, input, dim0, dim1):
        if hasattr(input, 'meta'):
            input = input.meta['val']
        shape = list(input.shape)
        (shape[dim1], shape[dim0]) = (shape[dim0], shape[dim1])
        return TensorInfo(shape, dtype=input.dtype, memory_format=get_memory_format(input))


class SoftmaxV2(Operator):
    def __init__(self):
        super().__init__("SoftmaxV2")
        self.torch_op = aten._softmax


class ReduceSumD(Operator):
    def __init__(self):
        super().__init__("ReduceSumD")
        self.torch_op = aten.sum


class Unsqueeze(Operator):
    def __init__(self):
        super().__init__("Unsqueeze")
        self.torch_op = aten.unsqueeze


class Squeeze(Operator):
    def __init__(self):
        super().__init__("Squeeze")
        # self.torch_op = aten.squeeze


class Pack(Operator):
    def __init__(self):
        super().__init__("Pack")


class Permute(Operator):
    def __init__(self):
        super().__init__("Permute")
        self.torch_op = aten.permute


class Expand(Operator):
    def __init__(self):
        super().__init__("Expand")


class ExpandD(Operator):
    def __init__(self):
        super().__init__("ExpandD")

    def __call__(self, x, dims):
        if hasattr(x, 'meta'):
            x = x.meta['val']
        dims = [dim.meta['val'] if hasattr(
            dim, 'meta') else dim for dim in dims]

        with x.fake_mode:
            if not negative_in_shape(dims):
                return aten.empty(dims, dtype=x.dtype)
            return x.expand(dims)


class Sort(Operator):
    def __init__(self):
        super().__init__("Sort")
        self.torch_op = aten.sort


class TopK(Operator):
    def __init__(self):
        super().__init__("TopK")
        self.torch_op = aten.topk


class ScatterElement(Operator):
    def __init__(self):
        super().__init__("ScatterElements")


class ReduceMean(Operator):
    def __init__(self):
        super().__init__("ReduceMean")
        self.torch_op = aten.mean


class ReduceStdV2Update(Operator):
    def __init__(self):
        super().__init__("ReduceStdV2Update")


class ReduceMaxD(Operator):
    def __init__(self):
        super().__init__("ReduceMaxD")
        self.torch_op = aten.amax


class Const(Operator):
    def __init__(self):
        super().__init__("Const")


class Sigmoid(Operator):
    def __init__(self):
        super().__init__("Sigmoid")
        self.torch_op = aten.sigmoid


class Pow(Operator):
    def __init__(self):
        super().__init__("Pow")
        self.torch_op = aten.pow


class Select(Operator):
    def __init__(self):
        super().__init__("Select")


class LessEqual(Operator):
    def __init__(self):
        super().__init__("LessEqual")
        self.torch_op = aten.le


class Less(Operator):
    def __init__(self):
        super().__init__("Less")


class Equal(Operator):
    def __init__(self):
        super().__init__("Equal")


class Conv2D(Operator):
    def __init__(self):
        super().__init__("Conv2D")
        self.torch_op = aten.convolution


class GreaterEqual(Operator):
    def __init__(self):
        super().__init__("GreaterEqual")


class InAdd(Operator):
    def __init__(self):
        super().__init__("inadd")
        self.torch_op = _operator.add


class Cast(Operator):
    def __init__(self):
        super().__init__("Cast")


class Identity(Operator):
    def __init__(self):
        super().__init__("Identity")


class IdentityN(Operator):
    def __init__(self):
        super().__init__("IdentityN")

    def __call__(self, *args, **kwargs):
        def get_meta(x):
            return x if not hasattr(x, 'meta') else x.meta['val']
        new_args = tree_map(get_meta, args)
        res = []
        for item in new_args:
            with item.fake_mode:
                res.append(aten.clone(item))
        return res


class ZerosLike(Operator):
    def __init__(self):
        super().__init__("ZerosLike")
        self.torch_op = aten.full_like


class Empty(Operator):
    def __init__(self):
        super().__init__("Empty")
        self.torch_op = aten.empty


class GatherV2(Operator):
    def __init__(self):
        super().__init__("GatherV2")


class OnesLike(Operator):
    def __init__(self):
        super().__init__("OnesLike")
        self.torch_op = aten.ones


class Fill(Operator):
    def __init__(self):
        super().__init__("Fill")


class Conv2DBackpropInput(Operator):
    def __init__(self):
        super().__init__("Conv2DBackpropInput")


class Conv2DBackpropFilter(Operator):
    def __init__(self):
        super().__init__("Conv2DBackpropFilter")


class LogSoftmaxV2(Operator):
    def __init__(self):
        super().__init__("LogSoftmaxV2")
        self.torch_op = aten._log_softmax


class LogSoftmaxGrad(Operator):
    def __init__(self):
        super().__init__("LogSoftmaxGrad")
        self.torch_op = aten._log_softmax_backward_data


class FillV2D(Operator):
    def __init__(self):
        super().__init__("FillV2D")


class NLLLoss(Operator):
    def __init__(self):
        super().__init__("NLLLoss")


class NLLLossGrad(Operator):
    def __init__(self):
        super().__init__("NLLLossGrad")


class BNTrainingReduce(Operator):
    def __init__(self):
        super().__init__("BNTrainingReduce")


class BNTrainingUpdate(Operator):
    def __init__(self):
        super().__init__("BNTrainingUpdate")


class BNTrainingUpdateGrad(Operator):
    def __init__(self):
        super().__init__("BNTrainingUpdateGrad")


class BNTrainingReduceGrad(Operator):
    def __init__(self):
        super().__init__("BNTrainingReduceGrad")


class ReluGrad(Operator):
    def __init__(self):
        super().__init__("ReluGrad")


class ThresholdGradV2D(Operator):
    def __init__(self):
        super().__init__("ThresholdGradV2D")


class ZerosLike(Operator):
    def __init__(self, x):
        super().__init__("ZerosLike")
        self.x = x
        self.torch_op = aten.zeros_like


class SplitD(Operator):
    def __init__(self):
        super().__init__("SplitD")


class Slice(Operator):
    def __init__(self):
        super().__init__("Slice")
        # self.torch_op = aten.slice


class ConcatD(Operator):
    def __init__(self):
        super().__init__("ConcatD")


class MaskedFill(Operator):
    def __init__(self):
        super().__init__("MaskedFill")
        self.torch_op = aten.masked_fill


class Reshape(Operator):
    def __init__(self):
        super().__init__("Reshape")


class Pad(Operator):
    def __init__(self):
        super().__init__("Pad")


class Fills(Operator):
    def __init__(self):
        super().__init__("Fills")


class SoftmaxGrad(Operator):
    def __init__(self):
        super().__init__("SoftmaxGrad")
        self.torch_op = aten._softmax_backward_data.default


class StatelessBernoulli(Operator):
    def __init__(self):
        super().__init__("StatelessBernoulli")
        # self.torch_op = aten.bernoulli.p


class Shape(Operator):
    def __init__(self):
        super().__init__("Shape")


class AddV2(Operator):
    def __init__(self):
        super().__init__("AddV2")


class StatelessRandomUniformV2(Operator):
    def __init__(self):
        super().__init__("StatelessRandomUniformV2")


class Greater(Operator):
    def __init__(self):
        super().__init__("Greater")


class Addcmul(Operator):
    def __init__(self):
        super().__init__("Addcmul")


class Reciprocal(Operator):
    def __init__(self):
        super().__init__("Reciprocal")
        self.torch_op = aten.reciprocal.default


def ret_triple(a, b, c) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return a, b, c


def ret_tuple(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b
