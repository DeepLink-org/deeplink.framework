import torch
from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer, SingleOpTransformer
from dicp.vendor.AscendGraph.ascend_op import MatMul
from dicp.vendor.AscendGraph.conversion import AtenToAscendTransformer
from dicp.dynamo_bridge.op_transformer import PatternMatcherPass
from dicp.vendor.AscendGraph.pattern_replacement import (
    aten_patterns_cls_list, 
    ascend_patterns_cls_list
)


# 该pass需要在FuseTransposeMatmul之后
class ArgsTransDataPass:
    def transform(self, gm:torch.fx.graph_module):
        for n in gm.graph.nodes:
            if n.op != 'call_function':
                continue
            if type(n.target) in [MatMul]:
                for arg in n.args:
                    if arg.op == 'placeholder':
                        arg.meta['format'] = 'FRACTAL_NZ'
        return gm


def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    gm = BackendPatternMatcherTransformer(
            PatternMatcherPass(), aten_patterns_cls_list).transform(gm)
    gm = AtenToAscendTransformer(gm).transform()
    gm = BackendPatternMatcherTransformer(
            PatternMatcherPass(), ascend_patterns_cls_list).transform(gm)
    # gm = ArgsTransDataPass().transform(gm)
    return gm
