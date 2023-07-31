import torch
import torch.fx

from dicp.dynamo_bridge.op_transformer import OpSetTransformer
from dicp.vendor.TopsGraph.conversion import patterns, conversions

def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, conversions).transform(gm)
