import torch
import torch.fx

from third_party.DICP.common.op_transformer import OpSetTransformer
from third_party.DICP.TopsGraph.conversion import patterns, conversions

def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, conversions).transform(gm)
