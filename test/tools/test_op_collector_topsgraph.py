import torch
from dicp.tools.op_collector import InnerCompilerOpCollectorContext


def bar(a, b):
    x = torch.abs(a)
    x = x + 1
    if x.sum() > 0:
        x = torch.rsqrt(x)
    return x * b


with InnerCompilerOpCollectorContext(
    inner_commpiler_func="dicp.dynamo_bridge.compile_fx.compile_fx_inner",
    compile_fx_func="dicp.dynamo_bridge.compile_fx.compile_fx",
    collector_name="demo",
    inner_compiler_param_key="inner_compile",
    write_file=True,
    bypass_graph_module=True,
) as ctx:
    opt_func = torch.compile(bar, backend='topsgraph')
    inputs_t0 = torch.randn(20, 4)
    inputs_t1 = torch.randn(20, 4)
    outputs_t = opt_func(inputs_t0, inputs_t1)
