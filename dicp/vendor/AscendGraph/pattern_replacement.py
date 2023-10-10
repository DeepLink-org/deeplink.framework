import functools
import torch
import dicp.vendor.AscendGraph.ascend_op as ascend_op
import dicp.vendor.AscendGraph.conversion as conversion
from dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    register_backend_patterns,
)

aten_patterns_cls_list = []
register_aten_pattern = functools.partial(register_backend_patterns, aten_patterns_cls_list)

ascend_patterns_cls_list = []
register_ascend_pattern = functools.partial(register_backend_patterns, ascend_patterns_cls_list)

@register_aten_pattern
class ReplaceVarMean(BackendPatternBase):
    def pattern(input, dims):
        return torch.ops.aten.var_mean.correction(input, dims, correction=0, keepdim=True)

    def replacement(input, dims):
        meanVal = torch.ops.aten.mean(input, dims, True)
        varVal = torch.ops.aten.var(input, dims, correction=1, keepdim=True)
        return ascend_op.ret_tuple(varVal, meanVal)


@register_ascend_pattern
class FuseBmmTransposeRhsPattern(BackendPatternBase):
    def pattern(x1, x2):
        transpose_3 = conversion.transpose(x2, 2, 3)
        expand_1 = conversion.expand(transpose_3, [1, 32, 128, 32])
        view_16 = conversion.view(expand_1, [32, 128, 32])
        return conversion.bmm(x1, view_16)

    def replacement(x1, x2):
        view_16 = conversion.view(x2, [32, 32, 128])
        return conversion.bmm(x1, view_16, adj_x1 = False, adj_x2 = True)


@register_ascend_pattern
class FuseMatMulTransePoseRhsPattern(BackendPatternBase):
        @staticmethod
        def pattern(x1, x2):
            t_1 = conversion.t(x2)
            return conversion.mm(x1, t_1)
        
        @staticmethod
        def replacement(x1, x2):
            return conversion.mm(x1, x2, trans_a = False, trans_b = True, change_input=False)
