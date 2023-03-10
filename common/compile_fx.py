import dataclasses
import functools
import itertools
import logging
import sys
import functorch
import torch.fx

from typing import List
from importlib import import_module
from torch._dynamo.utils import fake_mode_from_tensors
from .graph import GraphConverter

log = logging.getLogger(__name__)

dynamo_logging = import_module(f"torch._dynamo.logging")
dynamo_utils = import_module(f"torch._dynamo.utils")

count_calls = dynamo_utils.count_calls

from torch._functorch.aot_autograd import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

@functools.lru_cache(None)
def _step_logger():
    return dynamo_logging.get_step_logger(log)

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

    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    _step_logger()(
        logging.INFO,
        f"{backend} compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    shape_env = _shape_env_from_inputs(example_inputs)
    fake_mode = fake_mode_from_tensors(example_inputs)

    gc = GraphConverter(gm, backend)
    gc.convert()
    compiled_fn = gc.compile_to_fn()

    # TODO need align inputs?

    _step_logger()(
        logging.INFO,
        f"{backend} compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # aot autograd needs to know to pass in inputs as a list
    compiled_fn._boxed_call = True
    return compiled_fn

_graph_counter = itertools.count(0)

def compile_fx(
    model_: torch.fx.GraphModule,
    example_inputs_: List[torch.Tensor],
    backend: str
):
    """Main entrypoint to a compile given FX graph"""
    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = True

    num_example_inputs = len(example_inputs_)

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        return compile_fx_inner(
            model,
            example_inputs,
            num_fixed=fixed,
            graph_id=graph_id,
            backend = backend,
        )

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return compile_fx_inner(
            model,
            example_inputs,
            num_fixed=fixed,
            is_backward=True,
            graph_id=graph_id,
            backend = backend,
        )

    from torch._inductor.decomposition import select_decomp_table
    return aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler
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

def _shape_env_from_inputs(inputs):
    shape_env = None
    fake_mode = fake_mode_from_tensors(inputs)

    # TODO(voz): It would be nice to enable this assert, but there are lots of tests that
    # pass in real inputs for now.
    # if len(inputs) > 0:
    # assert fake_mode is not None, breakpoint()

    if fake_mode is not None:
        return fake_mode.shape_env

    # TODO(voz): Should we always have one anyway?
    return None
