import torch
from dicp.dynamo_bridge.op_transformer import OpSetTransformer, SingleOpTransformer
from dicp.vendor.AscendGraph.ascend_op import MatMul
from dicp.vendor.AscendGraph.conversion import conversions
from dicp.vendor.AscendGraph.pattern_replacement import aten_patterns_cls_list, ascend_patterns_cls_list


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
    gm = OpSetTransformer(aten_patterns_cls_list).transform(gm)
    gm = SingleOpTransformer(gm, conversions).transform()
    # gm = ArgsTransDataPass().transform(gm)
    gm = OpSetTransformer(ascend_patterns_cls_list).transform(gm)
    return gm
