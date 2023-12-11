import torch
from typing import Tuple
from dicp.dynamo_bridge.operator import Operator
from dicp.vendor.AscendGraph.infer_res_utils import *

from dicp.dynamo_bridge.utils import get_memory_format

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

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class Add(Operator):
    def __init__(self):
        super().__init__("add")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class BroadcastTo(Operator):
    def __init__(self):
        super().__init__("BroadcastTo")

    def infer_result(self, x, shape):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        shape, shape_shape, shape_dim, shape_dtype = get_fake_tensor_meta_val(shape)
        shape = shape_shape
        dims = zip(reversed(shape), reversed(x_shape))

        for i, t in enumerate(dims):
            tar_dim, cur_dim = t
            if tar_dim == -1:
                shape[-(i + 1)] = cur_dim
                continue
            elif cur_dim == 1:
                continue
            assert cur_dim == tar_dim, self.__class__.__name__ + ": shape mismatch!"

        # broadcast keep get_memory_format
        return torch.empty(shape, dtype=x_dtype, memory_format=get_memory_format(x))


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

    def infer_result(self, x1, x2, adj_x1=False, adj_x2=False):
        x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1)
        x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2)

        assert x1_dim == 3 and x2_dim == 3, (
            self.__class__.__name__ + ": bmm's inputs must be  3D tensor!"
        )  # no broadcast
        assert x1_dtype == x2_dtype, (
            self.__class__.__name__ + ": expect same input type!"
        )  # no dtype cast

        adj_x1_shape = (
            [x1.shape[0]] + list(reversed(x1.shape[1:])) if adj_x1 else list(x1.shape)
        )
        adj_x2_shape = (
            [x2.shape[0]] + list(reversed(x2.shape[1:])) if adj_x2 else list(x2.shape)
        )

        assert adj_x1_shape[2] == adj_x2_shape[1], (
            self.__class__.__name__ + ": shape mismatch!"
        )
        out_shape = adj_x1_shape[0:2] + [adj_x2_shape[2]]

        return torch.empty(
            out_shape, dtype=x1_dtype, memory_format=get_memory_format(x1)
        )


class Sub(Operator):
    def __init__(self):
        super().__init__("Sub")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class Mul(Operator):
    def __init__(self):
        super().__init__("Mul")
        self.torch_op = aten.mul

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class Div(Operator):
    def __init__(self):
        super().__init__("Div")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class DivNoNan(Operator):
    def __init__(self):
        super().__init__("DivNoNan")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class Maximum(Operator):
    def __init__(self):
        super().__init__("Maximum")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


class Rsqrt(Operator):
    def __init__(self):
        super().__init__("Rsqrt")

    def infer_result(self, x):
        return common_unary_op_infer(x)


class Sqrt(Operator):
    def __init__(self):
        super().__init__("Sqrt")

    def infer_result(self, x):
        return common_unary_op_infer(x)


class Log(Operator):
    def __init__(self):
        super().__init__("Log")

    def infer_result(self, x):
        return common_unary_op_infer(x)


class Exp(Operator):
    def __init__(self):
        super().__init__("Exp")

    def infer_result(self, x, base=-1.0, scale=1.0, shift=0.0):
        return common_unary_op_infer(x)


class Neg(Operator):
    def __init__(self):
        super().__init__("Neg")

    def infer_result(self, x, base=-1.0, scale=1.0, shift=0.0):
        return common_unary_op_infer(x)


class Relu(Operator):
    def __init__(self):
        super().__init__("Relu")

    def infer_result(self, x, base=-1.0, scale=1.0, shift=0.0):
        return common_unary_op_infer(x)


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
        return common_unary_op_infer(x)


class ReduceSumD(Operator):
    def __init__(self):
        super().__init__("ReduceSumD")

    def infer_result(self, x, dims, keepdim):
        return reduce_op_infer(x, dims, keepdim)


class Unsqueeze(Operator):
    def __init__(self):
        super().__init__("Unsqueeze")

    def infer_result(self, x, dim=None):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        assert dim is not None, (
            self.__class__.__name__ + ": doesn't specify axis to unsqueeze!"
        )
        x_shape = list(x_shape)
        for d in sorted(dim, reverse=True):
            x_shape.insert(d + x_dim + 1 if d < 0 else d, 1)
        return torch.empty(x_shape, dtype=x_dtype, memory_format=get_memory_format(x))


class Squeeze(Operator):
    def __init__(self):
        super().__init__("Squeeze")

    def infer_result(self, x, dim=None):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        if dim is None:
            shape = [i for i in x_shape if i != 1]
        else:
            shape = list(x_shape)
            for i in dim:
                assert x_shape[i] == 1, (
                    self.__class__.__name__
                    + ": can only squeeze a dimension that is 1!"
                )
                shape.pop(i)
        x_memory_format = get_memory_format(x)
        if len(shape) < 4:
            x_memory_format = torch.contiguous_format
        return torch.empty(shape, dtype=x_dtype, memory_format=x_memory_format)


class Pack(Operator):
    def __init__(self):
        super().__init__("Pack")

    def infer_result(self, x, dim):
        x0, x0_shape, x0_dim, x0_dtype = get_fake_tensor_meta_val(x[0])
        dim = (dim + x0_dim + 1) % (x0_dim + 1)
        out_shape = list(x0_shape)
        out_shape.insert(dim, len(x))
        return torch.empty(
            out_shape, dtype=x0_dtype, memory_format=get_memory_format(x0)
        )


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
        return reduce_op_infer(x, dims, keepdim)


class Const(Operator):
    def __init__(self):
        super().__init__("Const")

    def infer_result(self, *new_args, **kwargs):
        return new_args, kwargs


class Sigmoid(Operator):
    def __init__(self):
        super().__init__("Sigmoid")

    def infer_result(self, x):
        return common_unary_op_infer(x)


class Pow(Operator):
    def __init__(self):
        super().__init__("Pow")

    def infer_result(self, base, expo):
        base, base_shape, base_dim, base_dtype = get_fake_tensor_meta_val(base)
        if isinstance(expo, Tuple):  # Const
            expo, expo_shape = get_op_const_arg_kwarg(expo)
            expo_dtype = type(expo[0]) if len(expo) > 0 else base_dtype
        else:  # fake Tensor
            expo, expo_shape, expo_dim, expo_dtype = get_fake_tensor_meta_val(expo)

        out_shape = get_broadcast_res_two_shape(base_shape, expo_shape)
        dtype = get_cast_dtype(base_dtype, expo_dtype)
        memory_format = get_memory_format(base)
        return torch.empty(out_shape, dtype=dtype, memory_format=memory_format)


class Select(Operator):
    def __init__(self):
        super().__init__("Select")

    def infer_result(self, condition, x1, x2):
        x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1)
        x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2)
        _, c_shape, _, _ = get_fake_tensor_meta_val(condition)
        out_shape = get_broadcast_res_two_shape(
            get_broadcast_res_two_shape(x1_shape, c_shape), x2_shape
        )
        dtype = get_cast_dtype(x1_dtype, x2_dtype)
        memory_format = get_memory_format(x1)
        return torch.empty(out_shape, dtype=dtype, memory_format=memory_format)


class LessEqual(Operator):
    def __init__(self):
        super().__init__("LessEqual")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2, torch.bool)


class Less(Operator):
    def __init__(self):
        super().__init__("Less")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2, torch.bool)


class Equal(Operator):
    def __init__(self):
        super().__init__("Equal")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2, torch.bool)


class Conv2D(Operator):
    def __init__(self):
        super().__init__("Conv2D")


class GreaterEqual(Operator):
    def __init__(self):
        super().__init__("GreaterEqual")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2, torch.bool)


class InAdd(Operator):
    def __init__(self):
        super().__init__("inadd")


class Cast(Operator):
    def __init__(self):
        super().__init__("Cast")

    def infer_result(self, x, dtype):
        return common_unary_op_infer(x, ascend_type_to_torch(dtype))


class CastToCpu(Operator):
    def __init__(self):
        super().__init__("CastToCpu")


class Identity(Operator):
    def __init__(self):
        super().__init__("Identity")

    def infer_result(self, x, idx=None):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        out_dtype = x_dtype
        if x_dtype == torch.complex64:  # for complex64
            out_shape = list(x_shape)
            if idx == 0 or idx == 1:
                out_dtype = torch.float32
                out_shape.append(1)
        else:
            out_shape = [x_shape[idx]] if idx is not None else list(x_shape)
        return torch.empty(
            out_shape, dtype=out_dtype, memory_format=get_memory_format(x)
        )


class IdentityInp(Operator):
    def __init__(self):
        super().__init__("IdentityInp")

    def infer_result(self, src, dst):
        src, src_shape, src_dim, src_dtype = get_fake_tensor_meta_val(src)
        dst, dst_shape, dst_dim, dst_dtype = get_fake_tensor_meta_val(dst)
        out_shape = get_broadcast_res_two_shape(src_shape, dst_shape)
        return torch.empty(
            out_shape, dtype=dst_dtype, memory_format=get_memory_format(dst)
        )


class IdentityN(Operator):
    def __init__(self):
        super().__init__("IdentityN")

    def infer_result(self, x):
        return common_unary_op_infer(x)


class Empty(Operator):
    def __init__(self):
        super().__init__("Empty")


class GatherV2(Operator):
    def __init__(self):
        super().__init__("GatherV2")

    def infer_result(self, x, index, axis):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        idx, idx_shape, idx_dim, idx_dtype = get_fake_tensor_meta_val(index)
        idx_shape = list(idx_shape)
        idx_shape.append(x_shape[-1])
        return torch.empty(idx_shape, dtype=x_dtype, memory_format=get_memory_format(x))


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

    def infer_result(self, x, dim):
        return common_unary_op_infer(x)


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

    def infer_result(self, x):
        return common_unary_op_infer(x)


class SplitD(Operator):
    def __init__(self):
        super().__init__("SplitD")

    # TODO: params of this op is unclear, usage in conversion.py seems to be different with definition in Huawei code
    # current implementation regards this op ONLY being used for "view_as_complex"
    def infer_result(self, x, out_dim, split_dim, num_split):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        split_dim = (split_dim + x_dim) % x_dim
        out_shape = list(x_shape)
        del out_shape[-1]
        return torch.empty(
            out_shape,
            # TODO: need more info to infer type!
            # if this op is used in other op's decomposition, we have no idea what dtype the output should be
            dtype=torch.complex64
            if num_split == 2 and x_dim - 1 == out_dim
            else x_dtype,
            memory_format=get_memory_format(x),
        )


class Slice(Operator):
    def __init__(self):
        super().__init__("Slice")

    def infer_result(self, x, offset, size):
        x, x_shape, _, x_dtype = get_fake_tensor_meta_val(x)
        new_shape, _ = get_op_const_arg_kwarg(size)
        offset, _ = get_op_const_arg_kwarg(offset)
        _, storage_offset = cal_stride_offset(new_shape, offset, x)
        res = torch.as_strided(x, new_shape, x.stride(), storage_offset)
        return res


class ConcatD(Operator):
    def __init__(self):
        super().__init__("ConcatD")

    # TODO:memory_format?
    def infer_result(self, x, dim=0):
        x0, x0_shape, x0_dim, x0_dtype = get_fake_tensor_meta_val(x[0])
        dim = (dim + x0_dim) % x0_dim
        out_shape = list(x0_shape)
        out_shape[dim] = 0
        for t in x:
            _, t, _, _ = get_fake_tensor_meta_val(t)
            out_shape[dim] += t[dim]
        return torch.empty(
            out_shape, dtype=x0_dtype, memory_format=get_memory_format(x0)
        )


class MaskedFill(Operator):
    def __init__(self):
        super().__init__("MaskedFill")


class Reshape(Operator):
    def __init__(self):
        super().__init__("Reshape")

    def infer_result(self, x, shape_const_op):
        x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
        re_shape, re_dim = get_op_const_arg_kwarg(shape_const_op)
        # check whether stride and storage_offset are manually specified
        # if so, x is from operators like "Slice", and the stride and storage_offset still need to modify here
        x_stride = list(x.stride())
        x_shape = list(x_shape)

        use_x_stride = False  # if set to False(during following loop,find out it's part of op "select"), res's stride should be that of [re_shape]
        for i in range(len(x_stride) - 2, -1, -1):
            if x_stride[i + 1] * x_shape[i + 1] != x_stride[i]:  # x must from "Slice"!
                del x_stride[i + 1]
                del x_shape[i + 1]
                use_x_stride = True
                break

        # TODO: there may be something left out...
        # if param dim==0 in operator "select", res's stride may be identical to the result of operator "reshape" to the same final shape,
        # only the storage_offset (in dim 0) is different, thus it can use x_storage_offset at any case
        x_storage_offset = x.storage_offset()
        res = torch.empty(re_shape, dtype=x_dtype, memory_format=get_memory_format(x))
        res = torch.as_strided(
            res, re_shape, x_stride if use_x_stride else res.stride(), x_storage_offset
        )
        return res


class Pad(Operator):
    def __init__(self):
        super().__init__("Pad")


class Fills(Operator):
    def __init__(self):
        super().__init__("Fills")

    def infer_result(self, x, value):
        return common_unary_op_infer(x)


class SoftmaxGrad(Operator):
    def __init__(self):
        super().__init__("SoftmaxGrad")


class StatelessBernoulli(Operator):
    def __init__(self):
        super().__init__("StatelessBernoulli")

    def infer_result(self, target, prob, seed, offset, dtype):
        return common_unary_op_infer(
            target, spec_dtype=dtype, spec_format=torch.contiguous_format
        )


class Shape(Operator):
    def __init__(self):
        super().__init__("Shape")

    def infer_result(self, x):
        return common_unary_op_infer(x, spec_format=torch.contiguous_format)


class AddV2(Operator):
    def __init__(self):
        super().__init__("AddV2")

    def infer_result(self, x1, x2):
        return common_binary_op_infer(x1, x2)


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
