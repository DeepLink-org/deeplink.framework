import torch
import torch.fx
import sys
import os

from dicp.common.op_transformer import OpSetTransformer
from dicp.AscendGraph.conversion import patterns, conversions


def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, conversions).transform(gm)

