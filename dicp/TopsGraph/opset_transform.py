import torch
import torch.fx

from dicp.common.op_transformer import OpSetTransformer
from dicp.TopsGraph.conversion import patterns, conversions

def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, conversions).transform(gm)
