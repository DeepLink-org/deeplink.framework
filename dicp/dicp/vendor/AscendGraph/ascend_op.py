import typing
import torch
from typing import Tuple
from dicp.dynamo_bridge.operator import Operator
import numpy as np
from collections.abc import Sequence
from dicp.vendor.AscendGraph.infer_res_utils import *

from dicp.dynamo_bridge.utils import TensorInfo, get_memory_format

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
        self.torch_op = aten.add

    def infer_result(self, x1, x2):
        x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1, True)
        x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2, True)
        memory_format = get_memory_format(x1)
        dtype = get_cast_dtype(x1_dtype, x2_dtype)
        out_shape = get_broadcast_res_two_shape(x1_shape, x2_shape)

        return TensorInfo(shape=out_shape, dtype=dtype, memory_format=memory_format)


class Add(Operator):
    def __init__(self):
        super().__init__("add")
        self.torch_op = aten.add

    def infer_result(self, x1, x2):
        x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1, True)
        x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2, True)
        memory_format = get_memory_format(x1)
        dtype = get_cast_dtype(x1_dtype, x2_dtype)
        out_shape = get_broadcast_res_two_shape(x1_shape, x2_shape)

        return TensorInfo(shape=out_shape, dtype=dtype, memory_format=memory_format)


class BroadcastTo(Operator):
    def __init__(self):
        super().__init__("BroadcastTo")


class Range(Operator):
    def __init__(self):
        super().__init__("Range")


class Cumsum(Operator):
    def __init__(self):
        super().__init__("Cumsum")


class MatMul(Operator):
    def __init__(self):
        super().__init__("MatMul")


class BatchMatMul(Operator):
    def __init__(self):
        super().__init__("BatchMatMul")


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")


class Mul(Operator):
    def __init__(self):
        super().__init__("Mul")
        self.torch_op = aten.mul

    def infer_result(self, x1, x2):
        x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1, True)
        x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2, True)

        out_shape = get_broadcast_res_two_shape(x1_shape, x2_shape)
        dtype = get_cast_dtype(x1_dtype, x2_dtype)
        memory_format = get_memory_format(x1)

        return TensorInfo(shape=out_shape, dtype=dtype, memory_format=memory_format)


class Div(Operator):
    def __init__(self):
        super().__init__("Div")


class DivNoNan(Operator):
    def __init__(self):
        super().__init__("DivNoNan")


class Maximum(Operator):
    def __init__(self):
        super().__init__("Maximum")


class Rsqrt(Operator):
    def __init__(self):
        super().__init__("Rsqrt")


class Sqrt(Operator):
    def __init__(self):
        super().__init__("Sqrt")


class Log(Operator):
    def __init__(self):
        super().__init__("Log")
        self.torch_op = aten.log

    def infer_result(self, x):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)

        return TensorInfo(
            list(x_shape), dtype=x_dtype, memory_format=get_memory_format(x)
        )


class Exp(Operator):
    def __init__(self):
        super().__init__("Exp")


class Neg(Operator):
    def __init__(self):
        super().__init__("Neg")


class Relu(Operator):
    def __init__(self):
        super().__init__("Relu")


class Swish(Operator):
    def __init__(self):
        super().__init__("Swish")


class Transpose(Operator):
    def __init__(self):
        super().__init__("Transpose")


class SoftmaxV2(Operator):
    def __init__(self):
        super().__init__("SoftmaxV2")

    def infer_result(self, x, axes=None):
        x, x_shape, _, x_dtype = get_fake_tensor_meta_val(x, True)

        return TensorInfo(
            list(x_shape), dtype=x_dtype, memory_format=get_memory_format(x)
        )


class ReduceSumD(Operator):
    def __init__(self):
        super().__init__("ReduceSumD")


class Unsqueeze(Operator):
    def __init__(self):
        super().__init__("Unsqueeze")


class Squeeze(Operator):
    def __init__(self):
        super().__init__("Squeeze")


class Pack(Operator):
    def __init__(self):
        super().__init__("Pack")


class Permute(Operator):
    def __init__(self):
        super().__init__("Permute")


class Expand(Operator):
    def __init__(self):
        super().__init__("Expand")


class ExpandD(Operator):
    def __init__(self):
        super().__init__("ExpandD")


class Sort(Operator):
    def __init__(self):
        super().__init__("Sort")


class TopK(Operator):
    def __init__(self):
        super().__init__("TopK")


class ScatterElements(Operator):
    def __init__(self):
        super().__init__("ScatterElements")


class ReduceMean(Operator):
    def __init__(self):
        super().__init__("ReduceMean")


class ReduceStdV2Update(Operator):
    def __init__(self):
        super().__init__("ReduceStdV2Update")


class ReduceMaxD(Operator):
    def __init__(self):
        super().__init__("ReduceMaxD")

    def infer_result(self, x, dims, keepdim):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)

        out_shape = reduce_ops_output_size(x_shape, x_dim, dims, keepdim)

        return TensorInfo(
            shape=out_shape, dtype=x_dtype, memory_format=get_memory_format(x)
        )

    # def


class Const(Operator):
    def __init__(self):
        super().__init__("Const")

    def infer_result(self, x, dtype, x_dim):
        return TensorInfo(x_dim, dtype=dtype, memory_format=torch.contiguous_format)

    # def


class Sigmoid(Operator):
    def __init__(self):
        super().__init__("Sigmoid")


class Pow(Operator):
    def __init__(self):
        super().__init__("Pow")


class Select(Operator):
    def __init__(self):
        super().__init__("Select")


class LessEqual(Operator):
    def __init__(self):
        super().__init__("LessEqual")


class Less(Operator):
    def __init__(self):
        super().__init__("Less")


class Equal(Operator):
    def __init__(self):
        super().__init__("Equal")


class Conv2D(Operator):
    def __init__(self):
        super().__init__("Conv2D")


class GreaterEqual(Operator):
    def __init__(self):
        super().__init__("GreaterEqual")


class InAdd(Operator):
    def __init__(self):
        super().__init__("inadd")


class Cast(Operator):
    def __init__(self):
        super().__init__("Cast")

    def infer_result(self, x, dtype):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)

        return TensorInfo(
            list(x_shape), dtype=dtype, memory_format=get_memory_format(x)
        )


class CastToCpu(Operator):
    def __init__(self):
        super().__init__("CastToCpu")


class Identity(Operator):
    def __init__(self):
        super().__init__("Identity")


class IdentityInp(Operator):
    def __init__(self):
        super().__init__("IdentityInp")


class IdentityN(Operator):
    def __init__(self):
        super().__init__("IdentityN")


class Empty(Operator):
    def __init__(self):
        super().__init__("Empty")


class GatherV2(Operator):
    def __init__(self):
        super().__init__("GatherV2")


class OnesLike(Operator):
    def __init__(self):
        super().__init__("OnesLike")


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
        self.torch_op = aten._log_softmax.default

    def infer_result(self, x, dim):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)

        return TensorInfo(
            list(x_shape), dtype=x_dtype, memory_format=get_memory_format(x)
        )


class LogSoftmaxGrad(Operator):
    def __init__(self):
        super().__init__("LogSoftmaxGrad")


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


class SplitD(Operator):
    def __init__(self):
        super().__init__("SplitD")


class Slice(Operator):
    def __init__(self):
        super().__init__("Slice")


class ConcatD(Operator):
    def __init__(self):
        super().__init__("ConcatD")


class MaskedFill(Operator):
    def __init__(self):
        super().__init__("MaskedFill")


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

    def infer_result(self, x1, x2):
        x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1, True)
        x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2, True)
        memory_format = get_memory_format(x1)
        dtype = get_cast_dtype(x1_dtype, x2_dtype)
        out_shape = get_broadcast_res_two_shape(x1_shape, x2_shape)

        return TensorInfo(shape=out_shape, dtype=dtype, memory_format=memory_format)


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


class DropOutGenMaskV4(Operator):
    def __init__(self):
        super().__init__("DropOutGenMaskV4")


class DropOutDoMaskV3(Operator):
    def __init__(self):
        super().__init__("DropOutDoMaskV3")


def ret_triple(a, b, c) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return a, b, c


def ret_tuple(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    return a, b
