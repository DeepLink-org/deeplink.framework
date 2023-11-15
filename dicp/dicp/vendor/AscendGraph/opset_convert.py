import torch
from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
from dicp.vendor.AscendGraph.ascend_op import MatMul, CastToCpu, IdentityInp
from dicp.vendor.AscendGraph.conversion import AtenToAscendTransformer
from dicp.vendor.AscendGraph.pattern_replacement import (
    ascend_pattern_matcher,
    aten_patterns_cls_list,
    ascend_patterns_cls_list
)


# 该pass需要在FuseTransposeMatmul之后
class ArgsTransDataPass:
    def transform(self, gm: torch.fx.graph_module):
        for n in gm.graph.nodes:
            if n.op != 'call_function':
                continue
            if type(n.target) in [MatMul]:
                for arg in n.args:
                    if arg.op == 'placeholder':
                        arg.meta['format'] = 'FRACTAL_NZ'
        return gm


class OutputMarkPass:
    def __init__(self):
        self.assign_args = []
        self.cpu_tensor = []

    def transform(self, gm: torch.fx.graph_module):
        # dynamic shape feature
        input_names = []
        for n in gm.graph.nodes:
            if n.op == 'placeholder':
                input_names.append(n.name)

        for n in gm.graph.nodes:
            if n.op != 'call_function':
                continue
            if type(n.target) == CastToCpu:
                self.cpu_tensor.append(n.name)
            elif type(n.target) == IdentityInp:
                if len(n.args) == 2 and n.args[1] is not None and str(n.args[1]) in input_names:
                    self.assign_args.append((n.name, input_names.index(str(n.args[1]))))

        for n in gm.graph.nodes:
            if n.op == 'call_function':
                prop = {}
                if n.name in self.cpu_tensor:
                    prop.update({'cpu_tensor' : n.name})
                if len(self.assign_args) > 0 and n.name in list(zip(*self.assign_args))[0]:
                    idx = list(zip(*self.assign_args))[0].index(n.name)
                    prop.update({'assign_args' : (self.assign_args[idx][0], self.assign_args[idx][1])})
                n.meta['prop'] = prop
        return gm


def symint_in_inputs(nodes):
    # dynamic shape feature
    for node in nodes:
        if node.op == 'placeholder':
            if hasattr(node, 'meta'):
                node = node.meta['val']
            if isinstance(node, torch.SymInt):
                return True
            if hasattr(node, 'shape'):
                for dim in node.shape:
                    if isinstance(dim, torch.SymInt):
                        return True
    return False

def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    gm = BackendPatternMatcherTransformer(
        ascend_pattern_matcher, aten_patterns_cls_list).transform(gm)
    gm = AtenToAscendTransformer(gm).transform()
    if not symint_in_inputs(list(gm.graph.nodes)):
        gm = BackendPatternMatcherTransformer(
            ascend_pattern_matcher, ascend_patterns_cls_list).transform(gm)
    gm = OutputMarkPass().transform(gm)
    # gm = ArgsTransDataPass().transform(gm)
    return gm
