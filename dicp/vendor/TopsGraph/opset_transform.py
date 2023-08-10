import torch
import torch.fx

from dicp.dynamo_bridge.op_transformer import OpSetTransformer, SingleOpTransformer
from dicp.vendor.TopsGraph.conversion import patterns, conversions

def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):
    gm = OpSetTransformer(patterns).transform(gm)
    return SingleOpTransformer(gm, conversions).transform()
