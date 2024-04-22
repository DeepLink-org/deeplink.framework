import torch

from dicp.dynamo_bridge.decompositions import get_decompositions


def get_decomp():
    aten = torch.ops.aten
    return get_decompositions(
        [
            aten.count_nonzero.default,
        ]
    )


decomp = get_decomp()
