import torch
import torch._dynamo as dynamo

from torch import Tensor

from dicp.dynamo_bridge.decompositions import register_decomposition_for_dicp, get_decompositions

# for lightllm rotary_emb
@torch._custom_op.impl.custom_op('ascend::lightllm_rotary_emb')
def lightllm_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    ...

@lightllm_rotary_emb.impl_abstract()
def lightllm_rotary_emb_abstract(x, cos, sin):
    return torch.empty_like(x)

@lightllm_rotary_emb.impl(['cpu', 'cuda'])
def lightllm_rotary_emb_impl(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0: dim // 2]
    x1 = x[:, :, dim // 2: dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)

