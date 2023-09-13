import torch

from torch._inductor.decomposition import decompositions

def get_decomp():
    decompositions.clear()
    return decompositions

decomp = get_decomp()
