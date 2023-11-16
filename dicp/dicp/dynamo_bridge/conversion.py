import functools
import torch
from dicp.dynamo_bridge.operator import Operator


def args_kwargs_unchange(args, kwargs):
    return args, kwargs


def register_conversion_impl(
    conversions: list, aten_fn, decomp_fn, process_args_kwargs_fn=None
):
    register_op_singleton_flag = isinstance(
        decomp_fn, type) and issubclass(decomp_fn, Operator)
    if register_op_singleton_flag:
        wrapped = (decomp_fn.get_singleton(),
                   args_kwargs_unchange if process_args_kwargs_fn is None else process_args_kwargs_fn)
    else:
        @functools.wraps(decomp_fn)
        def wrapped(*args, **kwargs):
            return decomp_fn(*args, **kwargs)

    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in conversions:
                    aten_fn.append(other_fn)

    conversions.update({fn: wrapped for fn in aten_fn})
    if register_op_singleton_flag:
        return wrapped[0]
    else:
        return wrapped
