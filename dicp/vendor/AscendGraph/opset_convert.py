from typing import Any, Dict, Tuple

import torch
import torch.fx

import torch.fx.traceback as fx_traceback
from torch.fx.node import Argument, Target
from torch.fx.proxy import Proxy
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from dicp.dynamo_bridge.op_transformer import OpSetTransformer, SingleOpTransformer
from dicp.vendor.AscendGraph.ascend_op import MatMul
from dicp.vendor.AscendGraph.conversion import patterns, conversions


ascend_fuse_passes = []

def register_ascend_fuse_pass(cls):
    global ascend_fuse_passes
    ascend_fuse_passes.append(cls())
    return cls

#@register_ascend_fuse_pass        
class FuseTransposeMatmul():
    class MatMulOperatorSupport(OperatorSupportBase):
        def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
            return node.op == "call_function" and node.target in [
                torch.ops.aten.t.default, torch.ops.aten.mm.default,
            ]

    def __init__(self):
        self.support_ops = self.MatMulOperatorSupport()

    def transform(self, graph):
        partitioner = CapabilityBasedPartitioner(graph, self.support_ops,
                                                allows_single_node_partition=False)
        partitions = partitioner.propose_partitions()
        return partitioner.fuse_partitions(partitions)
    
    def hit(self, sub_nodes):
        match_trans = False
        match_mm = False
        for node in sub_nodes:
            if node.op == 'call_function' and node.target.name() == "aten::mm":
                match_mm = True
            if node.op == 'call_function' and node.target.name() == "aten::t":
                match_trans = True
        
        if match_trans and match_mm:
            return True
        return False
      
    def call_module(self, sub_nodes, args, kwargs):
        a = None
        b = None
        trans_a = False
        trans_b = False
        change_input = False

        input_map = {}
        for node in sub_nodes:
            if node.op == 'placeholder':
                input_map[node] = {
                    "node": args[len(input_map)],
                    "index": len(input_map),
                }
            if node.op == 'call_function' and node.target.name() == "aten::mm":
                (a, b) = node.args
                if a.op == 'call_function' and a.target.name() == "aten::t":
                    trans_a = True
                    real_args = input_map[a.args[0]]
                    change_input = True if real_args["index"] != 0 else False
                    a = real_args["node"]
                else:
                    real_args = input_map[a]
                    change_input = True if real_args["index"] != 0 else False
                    a = real_args["node"]
                if b.op == 'call_function' and b.target.name() == "aten::t":
                    trans_b = True
                    b = input_map[b.args[0]]["node"]
                else:
                    b = input_map[b]["node"]
        return MatMul(a, b, trans_a=trans_a, trans_b=trans_b,
                      change_input=change_input)


class AscendOpTransformer(SingleOpTransformer):
    def __init__(self, module, conversions):
        super().__init__(module, conversions)

    def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        sub_graph = self.fetch_attr(target)
        sub_nodes = [x for x in sub_graph.graph.nodes]
        
        out = None
        for p in ascend_fuse_passes:
            if p.hit(sub_nodes):
                out = p.call_module(sub_nodes, args, kwargs)
                break
        
        assert out is not None
        proxy = self.tracer.create_proxy('call_function', out, args, kwargs)
        proxy.node.meta = fx_traceback.get_current_meta()
        return proxy         


def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    for p in ascend_fuse_passes:
        gm = p.transform(gm)

    gm = OpSetTransformer(patterns).transform(gm)
    gm = AscendOpTransformer(gm, conversions).transform()
    return gm
