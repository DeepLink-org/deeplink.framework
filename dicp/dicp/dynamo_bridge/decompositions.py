from collections import defaultdict
from typing import Callable, Dict, Sequence, Union

import torch
from torch._decomp import register_decomposition
from torch._ops import OpOverload, OpOverloadPacket

dicp_decomposition_table = {}
aten = torch.ops.aten


def register_decomposition_for_dicp(fn):
    return register_decomposition(fn, registry=dicp_decomposition_table)


@register_decomposition_for_dicp(aten.count_nonzero.default)
def count_nonzero_default(x, dim=None):
    cond = x != 0
    dim = [] if dim is None else dim
    return aten.sum.dim_IntList(cond, dim=dim, keepdim=False, dtype=torch.int64)


def get_decompositions(
    aten_ops: Sequence[Union[OpOverload, OpOverloadPacket]],
    target_decomposition_table: Dict[OpOverload, Callable] = None,
) -> Dict[OpOverload, Callable]:
    registry = dicp_decomposition_table
    packets_to_overloads = defaultdict(list)
    for opo in registry:
        packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions = target_decomposition_table if target_decomposition_table else {} 
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = registry[op_overload]
        elif isinstance(op, OpOverload) and op in registry:
            decompositions[op] = registry[op]
    return decompositions
