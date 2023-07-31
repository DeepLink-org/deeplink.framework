import torch
import torch.fx
import sys
import os

from dicp.dynamo_bridge.op_transformer import OpSetTransformer
from dicp.vendor.AscendGraph.conversion import patterns, conversions


def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, conversions).transform(gm)

