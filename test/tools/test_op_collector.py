import torch
from third_party.DICP.tools.op_collector import InnerCompilerOpCollectorContext


def xyxy2xywh(boxes):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h



with InnerCompilerOpCollectorContext(
    inner_commpiler_func="torch._inductor.compile_fx.compile_fx_inner",
    compile_fx_func="torch._inductor.compile_fx.compile_fx",
    collector_name="resnet",
    inner_compiler_param_key="inner_compile",
    write_file=True,
    bypass_graph_module=True,
) as ctx:
    opt_func = torch.compile(xyxy2xywh)
    inputs_t = torch.randn(20, 4)
    outputs_t = opt_func(inputs_t)
    print(outputs_t)
    print(opt_func)
