import logging
import operator
import os
import re
import sys
import time
import torch
import torch.fx

from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import dynamo_timed

log = logging.getLogger(__name__)

class GraphTransformer:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        backend: str,
    ):
        self.gm = gm
        if backend == 'topsgraph':
            from dicp.TopsGraph.opset_transform import topsgraph_opset_transform
            self.backend_opset_transform = topsgraph_opset_transform
            from dicp.TopsGraph.codegen.enflame import EnflameCodegen
            self.backend_codegen = EnflameCodegen
        elif backend == 'ascendgraph':
            from dicp.AscendGraph.opset_convert import ascendgraph_opset_convert
            self.backend_opset_transform = ascendgraph_opset_convert
            from dicp.AscendGraph.codegen.ascend import AscendCodegen
            self.backend_codegen = AscendCodegen

    def transform(self):
        self.gm = self.backend_opset_transform(self.gm)

    def infer_shape_dtype(self):
        for n in self.gm.graph.nodes:
            if n.op == 'call_function':
                n.meta['val'] = (n.target(*n.args, **n.kwargs))

    def codegen(self):
        return self.backend_codegen(self.gm).codegen()

    @dynamo_timed
    def compile_to_module(self):
        from torch._inductor.codecache import PyCodeCache

        code = self.codegen()

        mod = PyCodeCache.load(code)

        # if dynamo_config.output_code:
        #     log.info("Output code: %s", mod.__file__)
        return mod

    def compile_to_fn(self):
        return self.compile_to_module().call
