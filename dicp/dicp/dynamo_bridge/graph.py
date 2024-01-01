import logging
import os
import torch
import torch.fx
from typing import Optional
from torch._dynamo.utils import dynamo_timed
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._inductor.codecache import cache_dir
from dicp.dynamo_bridge.utils import save_cpu_gm, AotOperatorUnsupport
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

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
        self.aot_operations = dict()
        self.aot_operations_prefix = ''
        if backend == 'topsgraph':
            from dicp.vendor.TopsGraph.opset_transform import topsgraph_opset_transform
            self.backend_opset_transform = topsgraph_opset_transform
            from dicp.vendor.TopsGraph.codegen.enflame import EnflameCodegen
            self.backend_codegen = EnflameCodegen
        elif backend == 'ascendgraph':
            from dicp.vendor.AscendGraph.opset_convert import ascendgraph_opset_convert
            self.backend_opset_transform = ascendgraph_opset_convert
            from dicp.vendor.AscendGraph.codegen.ascend import AscendCodegen
            self.backend_codegen = AscendCodegen
            from dicp.vendor.AscendGraph.config import enable_aot_operations, \
                aot_operations, aot_operations_prefix
            if enable_aot_operations:
                self.aot_operations = aot_operations
                self.aot_operations_prefix = aot_operations_prefix

    def transform(self):
        self.gm = self.backend_opset_transform(self.gm)

    def infer_shape_dtype(self):
        def make_tensor_meta(x) -> Optional[TensorMetadata]:
            if isinstance(x, FakeTensor):
                return _extract_tensor_metadata(x)
            else:
                return None
        test_infer = bool(os.environ.get("TEST_DICP_INFER", False))
        for n in self.gm.graph.nodes:
            fake_value = None
            if n.op == 'call_function':
                fake_value = (n.target(*n.args, **n.kwargs))
            elif n.op == 'get_attr':
                target_atoms = n.target.split('.')
                attr_itr = self.gm
                for i, atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(
                            f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                    attr_size, attr_dtye = attr_itr.shape, attr_itr.dtype
                with FakeTensorMode():
                    fake_value = torch.empty(attr_size, dtype=attr_dtye)
            else:
                continue
            if 'val' in n.meta and test_infer:
                assert n.meta['val'].size() == fake_value.size(), "check infer size failed"
                assert n.meta['val'].dtype == fake_value.dtype, "check infer dtype failed"
                assert n.meta['val'].stride() == fake_value.stride(), "check infer stride failed"
                assert n.meta['val'].storage_offset() == fake_value.storage_offset(), "check infer storage offset failed"
            if 'val' not in n.meta:
                n.meta['val'] = fake_value
                n.meta["tensor_meta"] = make_tensor_meta(n.meta['val'])

    def codegen(self):
        return self.backend_codegen(self.gm, self.cpu_gm, self.folder, self.graph_key).codegen()

    def partition_graph_with_aot_op(self):
        if not self.aot_operations:
            return
        operator_support = AotOperatorUnsupport(self.aot_operations, self.aot_operations_prefix)
        partitioner = CapabilityBasedPartitioner(self.gm, operator_support,
                                                 allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()
        fused_graph = partitioner.fuse_partitions(partitions)
        self.gm = fused_graph

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
