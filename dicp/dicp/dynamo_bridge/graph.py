import logging
import os
import torch
import torch.fx
from typing import List, Optional, Tuple
from torch._dynamo.utils import dynamo_timed
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._inductor.codecache import cache_dir
from dicp.dynamo_bridge.utils import save_cpu_gm
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

log = logging.getLogger(__name__)


class GraphTransformer:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        backend: str,
    ):
        self.gm = gm
        self.backend = backend
        self.folder = cache_dir()
        self.cpu_gm, self.graph_key = save_cpu_gm(gm, self.folder)
        if backend == 'topsgraph':
            from dicp.vendor.TopsGraph.opset_transform import topsgraph_opset_transform
            self.backend_opset_transform = topsgraph_opset_transform
            from dicp.vendor.TopsGraph.opset_transform import topsgraph_infer_shape
            self.backend_infer_shape = topsgraph_infer_shape
            from dicp.vendor.TopsGraph.codegen.enflame import EnflameCodegen
            self.backend_codegen = EnflameCodegen
        elif backend == 'ascendgraph':
            from dicp.vendor.AscendGraph.opset_convert import ascendgraph_opset_convert
            self.backend_opset_transform = ascendgraph_opset_convert
            from dicp.vendor.AscendGraph.opset_convert import ascendgraph_infer_shape
            self.backend_infer_shape = ascendgraph_infer_shape
            from dicp.vendor.AscendGraph.codegen.ascend import AscendCodegen
            self.backend_codegen = AscendCodegen

    def transform(self):
        self.gm = self.backend_opset_transform(self.gm)

    def infer_shape_dtype(self):
        self.gm = self.backend_infer_shape(self.gm)

    def codegen(self):
        return self.backend_codegen(self.gm, self.cpu_gm, self.folder, self.graph_key).codegen()

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
