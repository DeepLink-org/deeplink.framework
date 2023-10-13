import torch
from typing import Optional, Tuple, List

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    # assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    
    freqs_cis_real: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = torch.view_as_complex(freqs_cis_real)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis)
    return xq_out

a = torch.randn([1, 32, 32, 128], dtype=torch.float16)
b = torch.randn([1, 32, 32, 128], dtype=torch.float16)
c = torch.randn([32, 64, 2], dtype=torch.float32)

opt_model = torch.compile(apply_rotary_emb, backend='ascendgraph')

opt_res = opt_model(a, c)
print(opt_res.shape)

res = apply_rotary_emb(a, c)
print(res.shape)
