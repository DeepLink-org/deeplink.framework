import functools
import torch
import dicp.vendor.AscendGraph.ascend_op as ascend_op
from dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)
from dicp.vendor.AscendGraph.config import enable_aot_operations

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


Const = torch.fx.wrap(ascend_op.Const.get_singleton())
Transpose = torch.fx.wrap(ascend_op.Transpose.get_singleton())
Identity = torch.fx.wrap(ascend_op.Identity.get_singleton())
Reshape = torch.fx.wrap(ascend_op.Reshape.get_singleton())
BatchMatMul = torch.fx.wrap(ascend_op.BatchMatMul.get_singleton())
Permute = torch.fx.wrap(ascend_op.Permute.get_singleton())
MatMul = torch.fx.wrap(ascend_op.MatMul.get_singleton())


@register_ascend_pattern
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
class FuseMatMulTransePoseRhsPattern(BackendPatternBase):
    @staticmethod
    def pattern(x1, x2):
        t_1 = Permute(x2, [1, 0])
        return MatMul(x1, t_1, False, False)

    @staticmethod
    def replacement(x1, x2):
        return MatMul(x1, x2, trans_x1=False, trans_x2=True)

if enable_aot_operations:
    DIOPIRMSNorm = torch.fx.wrap(ascend_op.DIOPIRMSNorm.get_singleton())

    @register_aten_pattern
    class RMSNormFloat16Input(BackendPatternBase):
        @staticmethod
        def pattern(input_x, weight, eps, _to_copy_1_device):
            _to_copy = torch.ops.aten._to_copy.default(input_x, dtype=torch.float32)
            pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
            mean = torch.ops.aten.mean.dim(pow_1, [-1], True)
            add = torch.ops.aten.add.Tensor(mean, eps)
            rsqrt = torch.ops.aten.rsqrt.default(add)
            mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt)
            _to_copy_1 = torch.ops.aten._to_copy.default(mul, dtype=torch.float16,
                                                         layout=torch.strided, device=_to_copy_1_device)
            mul_1 = torch.ops.aten.mul.Tensor(_to_copy_1, weight)
            return mul_1

        @staticmethod
        def replacement(input_x, weight, eps):
            return DIOPIRMSNorm(input_x, weight, eps)
