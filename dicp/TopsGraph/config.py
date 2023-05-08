import torch

from torch._inductor.decomposition import decompositions

aten = torch.ops.aten
decomp_del_keys = [aten._native_batch_norm_legit_functional.default,
                   aten.native_batch_norm_backward.default,
                   aten.convolution_backward.default]
def get_decomp():
    for del_key in decomp_del_keys:
        if del_key in decompositions:
            del decompositions[del_key]
    return decompositions

decomp = get_decomp()
