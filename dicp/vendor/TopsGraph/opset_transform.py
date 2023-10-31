import torch
import torch.fx

from dicp.vendor.TopsGraph.conversion import AtenToTopsTransformer
from dicp.dynamo_bridge.compile_fx import is_torch_210

if is_torch_210:
    from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
    from dicp.vendor.TopsGraph.conversion import tops_patterns, aten_patterns_cls_list, tops_patterns_cls_list


def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):

    # 1aten to Naten
    if is_torch_210:
        gm = BackendPatternMatcherTransformer(
            tops_patterns, aten_patterns_cls_list).transform(gm)

    # 1aten to Ntops
    gm = AtenToTopsTransformer(gm).transform()

    # Ntops to Mtops
    if is_torch_210:
        gm = BackendPatternMatcherTransformer(
            tops_patterns, tops_patterns_cls_list).transform(gm)

    return gm
