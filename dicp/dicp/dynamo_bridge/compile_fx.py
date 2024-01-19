from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from .graph import GraphTransformer
import functools
import itertools
import logging
import sys
import functorch
import torch.fx
import importlib
import os
from dicp.dynamo_bridge import pt_patch # noqa F401

from typing import List
from importlib import import_module

import torch
is_torch_200 = False
is_torch_210 = False
if torch.__version__.startswith("2.0"):
    is_torch_200 = True
elif torch.__version__.startswith("2.1"):
    is_torch_210 = True
else:
    raise ValueError(f"unsupported dicp torch version: {torch.__version__}")


log = logging.getLogger(__name__)

dynamo_logging = import_module("torch._dynamo.logging")
dynamo_utils = import_module("torch._dynamo.utils")

count_calls = dynamo_utils.count_calls


def get_fake_mode_from_tensors(input_tensors):
    if is_torch_200:
        from torch._dynamo.utils import fake_mode_from_tensors
        return fake_mode_from_tensors(input_tensors)
    elif is_torch_210:
        from torch._dynamo.utils import detect_fake_mode
        return detect_fake_mode(input_tensors)
    else:
        raise ValueError(
            f"unsupported dicp torch version: {torch.__version__}")


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


@torch.utils._python_dispatch._disable_current_modes()
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    num_fixed=0,
    is_backward=False,
    graph_id=None,
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
    gt.partition_graph_with_aot_op()
    compiled_fn = gt.compile_to_fn()

    # aot autograd needs to know to pass in inputs as a list
    compiled_fn._boxed_call = True
    return compiled_fn


_graph_counter = itertools.count(0)


def compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    backend: str,
    inner_compile=compile_fx_inner,
):
    if torch.__version__.startswith("2.0"):
        return compile_fx_200(model_, example_inputs_, backend, inner_compile)
    elif torch.__version__.startswith("2.1"):
        return compile_fx_210(model_, example_inputs_, backend, inner_compile)
    else:
        raise ValueError(
            f"unsupported dicp torch version: {torch.__version__}")


def compile_fx_200(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    backend: str,
    inner_compile=compile_fx_inner,
):
    """Main entrypoint to a compile given FX graph"""
    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = True

    num_example_inputs = len(example_inputs_)

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            graph_id=graph_id,
            backend=backend,
        )

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            is_backward=True,
            graph_id=graph_id,
            backend=backend,
        )

    decompositions = get_decompositions(backend=backend)
    return aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        decompositions=decompositions
    )(model_, example_inputs_)


def compile_fx_210(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    backend: str,
    inner_compile=compile_fx_inner,
):
    import torch._dynamo.config as dynamo_config
    from torch._inductor.compile_fx import flatten_graph_inputs, graph_returns_tuple, \
        make_graph_return_tuple, pre_grad_passes, joint_graph_passes, min_cut_rematerialization_partition, \
        _PyTreeCodeGen, handle_dynamo_export_graph

    decompositions = get_decompositions(backend=backend)

    recursive_compile_fx = functools.partial(
        compile_fx,
        inner_compile=inner_compile,
        decompositions=decompositions,
    )

    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(
            model_,
            example_inputs_,
            recursive_compile_fx,
        )

    if isinstance(model_, torch.fx.GraphModule):
        if isinstance(model_.graph._codegen, _PyTreeCodeGen):
            # this graph is the result of dynamo.export()
            return handle_dynamo_export_graph(
                model_,
                example_inputs_,
                recursive_compile_fx,
            )

        # Since handle_dynamo_export_graph will trigger compile_fx again,
        # Move these passes after handle_dynamo_export_graph to avoid repeated calls.
        model_ = pre_grad_passes(model_, example_inputs_)

    if any(isinstance(x, (list, tuple, dict)) for x in example_inputs_):
        return flatten_graph_inputs(
            model_,
            example_inputs_,
            recursive_compile_fx,
        )

    # assert not config._raise_error_for_testing
    num_example_inputs = len(example_inputs_)

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler_base(model: torch.fx.GraphModule, example_inputs, is_inference):
        if is_inference:
            # partition_fn won't be called
            joint_graph_passes(model)

        fixed = len(example_inputs) - num_example_inputs
        return inner_compile(
            model,
            example_inputs,
            num_fixed=fixed,
            graph_id=graph_id,
            backend=backend,
        )

    fw_compiler = functools.partial(fw_compiler_base, is_inference=False)
    inference_compiler = functools.partial(fw_compiler_base, is_inference=True)

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
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        with dynamo_config.patch(dynamic_shapes=dynamic_shapes):
            fixed = count_tangents(model)
            return inner_compile(
                model,
                example_inputs,
                num_fixed=fixed,
                is_backward=True,
                graph_id=graph_id,
                backend=backend,
            )

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


def count_tangents(fx_g: torch.fx.GraphModule):
    """
    Infers which inputs are static for a backwards graph
    """

    def is_not_gradout(x):
        return "tangents" not in x.name

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_not_gradout(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)


def get_decompositions(backend):
    decompositions = {}
    folder_list = os.listdir(os.path.dirname(
        os.path.dirname(__file__)) + '/vendor')
    found_decomp = False
    for folder in folder_list:
        if backend.lower() == folder.lower():
            config = importlib.import_module(
                "dicp.vendor." + folder + ".config")
            decompositions = config.decomp
            found_decomp = True
    assert found_decomp, "Not found decomp table!"
    return decompositions
