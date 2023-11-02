import torch
import _operator
from typing import Tuple
from torch.utils._pytree import tree_map
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


class BroadcastTo(Operator):
    def __init__(self):
        super().__init__("BroadcastTo")


class Range(Operator):
    def __init__(self):
        super().__init__("Range")


class Cumsum(Operator):
    def __init__(self):
        super().__init__("Cumsum")
        self.torch_op = aten.cumsum.default


class MatMul(Operator):
    def __init__(self):
        super().__init__("MatMul")
        self.torch_op = aten.mm


class BatchMatMul(Operator):
    def __init__(self):
        super().__init__("BatchMatMul")
        self.torch_op = aten.bmm


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")


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


class ScatterElements(Operator):
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


class Empty(Operator):
    def __init__(self):
        super().__init__("Empty")


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
