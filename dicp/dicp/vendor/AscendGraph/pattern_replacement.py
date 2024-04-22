import functools
import torch
import dicp.vendor.AscendGraph.ascend_op as ascend_op
from dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)
ascend_pattern_matcher = PatternMatcherPass()

aten_patterns_cls_list = []
register_aten_pattern = functools.partial(
    register_backend_patterns, aten_patterns_cls_list)

ascend_patterns_cls_list = []
register_ascend_pattern = functools.partial(
    register_backend_patterns, ascend_patterns_cls_list)


@register_aten_pattern
class ReplaceVarMean(BackendPatternBase):
    def pattern(input, dims):
        return torch.ops.aten.var_mean.correction(input, dims, correction=0, keepdim=True)

    def replacement(input, dims):
        meanVal = torch.ops.aten.mean(input, dims, True)
        varVal = torch.ops.aten.var(input, dims, correction=1, keepdim=True)
        return ascend_op.ret_tuple(varVal, meanVal)


@register_aten_pattern
class FusedRepeatInterleaveSelfInt(BackendPatternBase):
    @staticmethod
    def pattern(self, repeat, dim, input_shape, empty_device, view_1_shape,
                expand_1_shape, repeat_interleave_output_size):
        empty = torch.ops.aten.empty.memory_format(input_shape, dtype=torch.int64, layout=torch.strided, device=empty_device)
        fill = torch.ops.aten.fill.Scalar(empty, repeat)
        view_1 = torch.ops.aten.view.default(fill, view_1_shape)
        expand_1 = torch.ops.aten.expand.default(view_1, expand_1_shape)
        repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(expand_1, output_size=repeat_interleave_output_size)
        index_select = torch.ops.aten.index_select.default(self, dim, repeat_interleave)
        return index_select

    @staticmethod
    def replacement(self, repeat, dim):
        return torch.ops.aten.repeat_interleave.self_int(self, repeat, dim)


@register_aten_pattern
class ReplaceAtenSliceScatter(BackendPatternBase):
    @staticmethod
    def pattern(arg0, arg1, start_index, end_index):
        slice = torch.ops.aten.slice.Tensor(arg0, 0, start_index, end_index)
        copy = torch.ops.aten.copy.default(slice, arg1)
        slice_scatter = torch.ops.aten.slice_scatter.default(arg0, copy, 0, start_index, end_index)
        copy_ = torch.ops.aten.copy_.default(slice_scatter, arg0)
        return slice_scatter

    @staticmethod
    def replacement(arg0, arg1, start_index, end_index):
        slice_scatter = torch.ops.lightllm.copy_with_offset.default(arg0, arg1, start_index, end_index)
        return slice_scatter


Muls = torch.fx.wrap(ascend_op.Muls.get_singleton())
Shape = torch.fx.wrap(ascend_op.Shape.get_singleton())
Const = torch.fx.wrap(ascend_op.Const.get_singleton())
Transpose = torch.fx.wrap(ascend_op.Transpose.get_singleton())
Identity = torch.fx.wrap(ascend_op.Identity.get_singleton())
Reshape = torch.fx.wrap(ascend_op.Reshape.get_singleton())
BatchMatMul = torch.fx.wrap(ascend_op.BatchMatMul.get_singleton())
Permute = torch.fx.wrap(ascend_op.Permute.get_singleton())
MatMul = torch.fx.wrap(ascend_op.MatMul.get_singleton())

Pow = torch.fx.wrap(ascend_op.Pow.get_singleton())
ReduceMeanD = torch.fx.wrap(ascend_op.ReduceMeanD.get_singleton())
Adds = torch.fx.wrap(ascend_op.Adds.get_singleton())
Rsqrt = torch.fx.wrap(ascend_op.Rsqrt.get_singleton())
ZerosLike = torch.fx.wrap(ascend_op.ZerosLike.get_singleton())
Less = torch.fx.wrap(ascend_op.Less.get_singleton())
Select = torch.fx.wrap(ascend_op.Select.get_singleton())
Mul = torch.fx.wrap(ascend_op.Mul.get_singleton())
Div = torch.fx.wrap(ascend_op.Div.get_singleton())
RmsNorm = torch.fx.wrap(ascend_op.RmsNorm.get_singleton())


# @register_ascend_pattern
class FuseBmmTransposeRhsPattern(BackendPatternBase):
    @staticmethod
    def pattern(x1, x2, dtype):
        const1 = Const([0, 1, 3, 2], dtype)
        transpose_2 = Transpose(x2, const1)
        identity1 = Identity(transpose_2, None)
        shape = Const([32, 128, 32], dtype)
        reshape = Reshape(identity1, shape)
        return BatchMatMul(x1, reshape, False, False)

    @staticmethod
    def replacement(x1, x2, dtype):
        shape = Const([32, 32, 128], dtype)
        reshape = Reshape(x2, shape)
        return BatchMatMul(x1, reshape, adj_x1=False, adj_x2=True)


@register_ascend_pattern
class FuseBmmTransposeMulsPattern(BackendPatternBase):
    @staticmethod
    def pattern(x1, x2, c1, c2):
        transpose = Transpose(x2, c1)
        muls = Muls(transpose, 0.3535533905932738)
        identity = Identity(muls, None)
        identity1 = Identity(identity, None)
        reshape = Reshape(identity1, c2)
        return BatchMatMul(x1, reshape, False, False, 0)

    @staticmethod
    def replacement(x1, x2, c1, c2):
        x2 = Reshape(x2, c2)
        perm = Permute(x2, [0, 2, 1])
        shape = Shape(perm)
        reshape = Reshape(x2, shape)
        muls = Muls(reshape, 0.3535533905932738)
        return BatchMatMul(x1, muls, adj_x1=False, adj_x2=True, keep_dtype=0)


@register_ascend_pattern
class FuseLightLLMRmsNorm(BackendPatternBase):
    @staticmethod
    def pattern(arg0_1, arg1_1):
        const = Const([2], torch.float32)
        pow_1 = Pow(arg0_1, const)
        reduce_mean_d = ReduceMeanD(pow_1, [-1], True, False)
        adds = Adds(reduce_mean_d, 0.001)
        rsqrt = Rsqrt(adds)
        zeros_like = ZerosLike(adds)
        div = Div(zeros_like, zeros_like)
        less = Less(adds, zeros_like)
        select = Select(less, div, rsqrt)
        mul = Mul(arg0_1, select)
        mul_1 = Mul(mul, arg1_1)
        return mul_1

    @staticmethod
    def replacement(arg0_1, arg1_1):
        rms_norm = RmsNorm(arg0_1, arg1_1, 0.001)
        return Identity(rms_norm, 0)


# @pandaoxin negotiate with @tangzhiyi
# another submit would implement
# @register_ascend_pattern
class FuseMatMulTransePoseRhsPattern(BackendPatternBase):
    @staticmethod
    def pattern(x1, x2):
        t_1 = Permute(x2, [1, 0])
        return MatMul(x1, t_1, False, False)

    @staticmethod
    def replacement(x1, x2):
        return MatMul(x1, x2, trans_x1=False, trans_x2=True)
