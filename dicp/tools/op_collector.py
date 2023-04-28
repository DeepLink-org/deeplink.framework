import importlib
from functools import wraps
from typing import Any

from torch.fx import Interpreter, GraphModule, Node
from torch.fx.node import _get_qualified_name
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch._subclasses.fake_tensor import FakeTensorMode


class OpCollector(Interpreter):
    def __init__(self, module: GraphModule,
                 garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.op_set = set()

    def run_node(self, n: Node) -> Any:
        if n.op in CALLABLE_NODE_OPS:
            if isinstance(n.target, str):
                self.op_set.add(n.target)
            else:
                self.op_set.add(_get_qualified_name(n.target))
        return super().run_node(n)

    def collect(self, *args):
        with FakeTensorMode() as mode:
            fake_args = [mode.from_tensor(a) for a in args]
            return super().run(*fake_args)

# Purpose:
# op collector context for inner compiler of dynamo backends
#
# Usage:
# with InnerCompilerOpCollectorContext(
#     inner_commpiler_func="torch._inductor.compile_fx.compile_fx_inner",
#     compile_fx_func="torch._inductor.compile_fx.compile_fx",
#     collector_name="resnet",
#     inner_compiler_param_key="inner_compile",
#     write_file=True,
#     bypass_graph_module=False,
# ) as ctx:
#     model = Model(param)
#     opt_model = torch.compile(model)
#     inputs = torch.randn(*shape)
#     loss = opt_model(inputs)
#     loss.backward()
class InnerCompilerOpCollectorContext:
    def __init__(self, inner_commpiler_func="torch._inductor.compile_fx.compile_fx_inner",
                 compile_fx_func="torch._inductor.compile_fx.compile_fx",
                 collector_name="module", inner_compiler_param_key="inner_compile",
                 write_file=False, bypass_graph_module=True):
        if isinstance(inner_commpiler_func, str):
            module_str, func_str = inner_commpiler_func.rsplit(".", 1)
            inner_commpiler_module = importlib.import_module(module_str)
            self.inner_commpiler_func = getattr(inner_commpiler_module, func_str)
        else:
            self.inner_commpiler_func = inner_commpiler_func
        if isinstance(compile_fx_func, str):
            module_str, func_str = compile_fx_func.rsplit(".", 1)
            compile_fx_module = importlib.import_module(module_str)
            self.compile_fx_func = getattr(compile_fx_module, func_str)
        else:
            self.compile_fx_func = compile_fx_func
        self.inner_compiler_param_key = inner_compiler_param_key
        self.collector_name = collector_name
        self.write_file = write_file
        self.bypass_graph_module = bypass_graph_module
        self.cached_gm_inputs_dict = None

    def __enter__(self):
        self.cached_gm_inputs_dict = dict()

        @wraps(self.inner_commpiler_func)
        def wrapped_inner_commpiler_func(gm, example_inputs, **kwargs):
            cache_idx = len(self.cached_gm_inputs_dict)
            self.cached_gm_inputs_dict[cache_idx] = (gm, tuple(example_inputs))
            if not self.bypass_graph_module:
                return self.inner_commpiler_func(gm, example_inputs, **kwargs)
            else:
                return gm

        @wraps(self.compile_fx_func)
        def wrapped_compile_fx_func(*args, **kwargs):
            kwargs[self.inner_compiler_param_key] = wrapped_inner_commpiler_func
            return self.compile_fx_func(*args, **kwargs)

        setattr(importlib.import_module(self.compile_fx_func.__module__),
                self.compile_fx_func.__name__,
                wrapped_compile_fx_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        op_dict_from_gm = dict()
        for key, v in self.cached_gm_inputs_dict.items():
            gm, inputs = v
            collector = OpCollector(gm)
            collector.collect(*inputs)
            op_dict_from_gm[key] = collector.op_set

        final_op_set = set()
        for k in op_dict_from_gm:
            final_op_set.update(op_dict_from_gm[k])
        final_op_list_sorted = sorted(final_op_set)
        for op in final_op_list_sorted:
            print(op)
        if self.write_file:
            output_file_name = f"{self.collector_name}_op.txt"
            with open(output_file_name, "w") as output_file:
                for op in final_op_list_sorted:
                    output_file.write(op + "\n")
            print(f"op collected in {output_file_name}")

        setattr(importlib.import_module(self.compile_fx_func.__module__),
                self.compile_fx_func.__name__,
                self.compile_fx_func)
