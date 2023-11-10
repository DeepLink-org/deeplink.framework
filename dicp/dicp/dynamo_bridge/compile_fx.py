from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from .graph import GraphTransformer
import functools
import logging
import sys
import torch.fx
import importlib
import os
from typing import List
from importlib import import_module
import torch

is_torch_210 = True if torch.__version__.startswith("2.1") else False
assert is_torch_210, f"unsupported dicp torch version: {torch.__version__}"

log = logging.getLogger(__name__)

dynamo_logging = import_module("torch._dynamo.logging")
dynamo_utils = import_module("torch._dynamo.utils")

count_calls = dynamo_utils.count_calls


def used_nodes_all_symint(nodes):
    for node in nodes:
        if node.op == 'placeholder' and len(node.users) > 0:
            if hasattr(node, 'meta'):
                node = node.meta['val']
            if not isinstance(node, torch.SymInt):
                return False
        elif node.op == 'output':
            if hasattr(node, 'meta') and 'val' in node.meta:
                node = node.meta['val']
            if not isinstance(node, torch.SymInt):
                return False
    return True


def get_decompositions(backend):
    decompositions = {}
    folder_list = os.listdir(os.path.dirname(os.path.dirname(__file__)) + '/vendor')
    found_decomp = False
    for folder in folder_list:
        if backend.lower() == folder.lower():
            config = importlib.import_module("dicp.vendor." + folder + ".config")
            decompositions = config.decomp
            found_decomp = True
    assert found_decomp, "Not found decomp table!"
    return decompositions


@torch.utils._python_dispatch._disable_current_modes()
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    backend=None
):
    if dynamo_utils.count_calls(gm.graph) == 0:
        return make_boxed_func(gm.forward)

    # all symint inputs fallback to eager mode
    if used_nodes_all_symint(list(gm.graph.nodes)):
        return gm

    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    gt = GraphTransformer(gm, backend)
    gt.transform()
    gt.infer_shape_dtype()
    compiled_fn = gt.compile_to_fn()

    # aot autograd needs to know to pass in inputs as a list
    compiled_fn._boxed_call = True
    return compiled_fn


def compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    backend: str,
    inner_compile=compile_fx_inner,
):
    import torch._dynamo.config as dynamo_config
    from torch._inductor.compile_fx import pre_grad_passes, joint_graph_passes, min_cut_rematerialization_partition

    decompositions = get_decompositions(backend=backend)

    # Since handle_dynamo_export_graph will trigger compile_fx again,
    # Move these passes after handle_dynamo_export_graph to avoid repeated calls.
    model_ = pre_grad_passes(model_, example_inputs_)

    def partition_fn(graph, joint_inputs, **kwargs):
        joint_graph_passes(graph)
        return min_cut_rematerialization_partition(
            graph, joint_inputs, **kwargs, compiler="inductor"
        )

    # Save and restore dynamic shapes setting for backwards, as it is
    # sometimes done as a context manager which won't be set when we
    # hit backwards compile
    dynamic_shapes = dynamo_config.dynamic_shapes

    @dynamo_utils.dynamo_timed
    def compiler_base(model: torch.fx.GraphModule, example_inputs, is_inference):
        if is_inference:
            joint_graph_passes(model)
        with dynamo_config.patch(dynamic_shapes=dynamic_shapes):
            return inner_compile(model, backend=backend)

    fw_compiler = functools.partial(compiler_base, is_inference=False)
    bw_compiler = functools.partial(compiler_base, is_inference=False)
    inference_compiler = functools.partial(compiler_base, is_inference=True)

    # TODO: can add logging before/after the call to create_aot_dispatcher_function
    # in torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
    # once torchdynamo is merged into pytorch
    return aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        decompositions=decompositions,
        partition_fn=partition_fn,
        keep_inference_input_mutations=True,
    )(model_, example_inputs_)
