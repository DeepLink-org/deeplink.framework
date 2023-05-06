import torch

from torch._inductor.decomposition import decompositions

def get_decomp():
    aten = torch.ops.aten
    del decompositions[aten._native_batch_norm_legit_functional.default]
    del decompositions[aten.native_batch_norm_backward.default]
    del decompositions[aten.convolution_backward.default]
    return decompositions

decomp = get_decomp()
