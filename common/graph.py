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

class GraphConverter:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        backend: str,
    ):
        self.gm = gm
        if backend == 'topsgraph':
            from TopsGraph.opset_convert import topsgraph_opset_convert
            self.backend_opset_convert = topsgraph_opset_convert
            from TopsGraph.codegen.enflame import EnflameCodegen
            self.backend_codegen = EnflameCodegen

    def convert(self):
        self.gm = self.backend_opset_convert(self.gm)

    def codegen(self):
        return self.backend_codegen(self.gm).codegen()

    @dynamo_timed
    def compile_to_module(self):
        from torch._inductor.codecache import PyCodeCache

        code = self.codegen()

        mod = PyCodeCache.load(code)
        for name, value in self.constants.items():
            setattr(mod, name, value)

        if dynamo_config.output_code:
            log.info("Output code: %s", mod.__file__)
        return mod

    def compile_to_fn(self):
        return self.compile_to_module().call
