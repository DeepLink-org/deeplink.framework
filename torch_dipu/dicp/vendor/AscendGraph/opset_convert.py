import torch
import torch.fx
import sys
import os

from torch_dipu.dicp.dynamo_bridge.op_transformer import OpSetTransformer
from torch_dipu.dicp.vendor.AscendGraph.conversion import patterns, conversions


def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, conversions).transform(gm)

