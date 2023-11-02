import torch

from torch._decomp import get_decompositions


aten = torch.ops.aten
decomp_keys = [aten.native_dropout.default,
               aten.native_dropout_backward.default]


def get_decomp():
    return get_decompositions(decomp_keys)


decomp = get_decomp()
