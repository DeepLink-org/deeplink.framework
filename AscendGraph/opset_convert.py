import torch
import torch.fx
from common.op_transformer import OpSetTransformer
from AscendGraph.conversion import patterns
from AscendGraph.op_whitelist import trans_namespace_ops

def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    return OpSetTransformer(patterns, "ascend", trans_namespace_ops).transform(gm)

