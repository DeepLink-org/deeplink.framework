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

class GraphTransformation:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
    ):
        # TODO add get_backend
        # self.get_backend(device).codegen_nodes(node.get_nodes())
        from .codegen.enflame import EnflameCodegen
        self.backend_codegen = EnflameCodegen(gm)

    def transform(self):
        pass

    def codegen(self):
        return self.backend_codegen.codegen()

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
