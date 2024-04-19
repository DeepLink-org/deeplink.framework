import math

import torch
import torch_dipu
import torch._dynamo as dynamo
import torch.nn.functional as F

from torch import Tensor
from typing import Sequence

torch._dynamo.config.suppress_errors = False

# rotary_emb
@torch._custom_op.impl.custom_op('lightllm::rotary_emb')
def rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    ...

@rotary_emb.impl_abstract()
def lightllm_rotary_emb_abstract(x, cos, sin):
    return torch.empty_like(x)

@rotary_emb.impl(['cpu', 'cuda'])
def lightllm_rotary_emb_impl(x, cos, sin):
    seq_len, h, dim = x.shape
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    x0 = x[:, :, 0: dim // 2]
    x1 = x[:, :, dim // 2: dim]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)

# rms_norm
@torch._custom_op.impl.custom_op('lightllm::rms_norm')
def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    ...

@rms_norm.impl_abstract()
def lightllm_rms_norm_abstract(x, weight, eps):
    return torch.empty_like(x)

@rms_norm.impl(['cpu', 'cuda'])
def lightllm_rms_norm_impl(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


@torch._custom_op.impl.custom_op('lightllm::prompt_attention_inference')
def prompt_attention_inference(q: Tensor, k: Tensor, v: Tensor, seqlen: Tensor, num_head: int, head_dim: int) -> Tensor:
    ...

@prompt_attention_inference.impl_abstract()
def lightllm_prompt_attention_inference_abstract(q: Tensor, k: Tensor, v: Tensor, seqlen: Tensor, num_head: int, head_dim: int):
    return torch.empty_like(q)

@prompt_attention_inference.impl(['cpu', 'cuda'])
def lightllm_prompt_attention_inference_impl(q, k, v, seqlen, num_head, head_dim):
    # prompt attention just support bs=1 for now.
    assert q.shape[0] == 1
    bs = q.shape[0]
    seqlen = seqlen.item()

    xq = q.view(bs, seqlen, num_head, head_dim)
    xk = k.view(bs, seqlen, num_head, head_dim)
    xv = v.view(bs, seqlen, num_head, head_dim)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0)
    mask = mask.masked_fill(mask == 0., -999999999999.0)
    mask = mask.masked_fill(mask == 1., 0.0)
    mask = mask.repeat(bs, num_head, 1, 1)

    keys = xk
    values = xv
    xq = xq.transpose(1, 2).float()
    keys = xk.transpose(1, 2).float()
    values = xv.transpose(1, 2).float()

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores + mask.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)

    return output

@torch._custom_op.impl.custom_op('lightllm::flash_attention_inference')
def flash_attention_inference(q: Tensor, all_k: Tensor, all_v: Tensor, currnet_lens: Sequence[int], max_len: int) -> Tensor:
    ...

@flash_attention_inference.impl_abstract()
def lightllm_flash_attention_inference_abstract(q: Tensor, all_k: Tensor, all_v: Tensor, currnet_lens: Sequence[int], max_len: int):
    return torch.empty_like(q)

@flash_attention_inference.impl(['cpu', 'cuda'])
def lightllm_flash_attention_inference_impl(q, all_k, all_v, current_lens, max_len):
    # q: batch, head, dim
    batch = q.shape[0]
    head = q.shape[1]
    dim = q.shape[2]
    res = []
    compute_batch = 1
    for i in range(batch):
        current_len = current_lens[i]
        kv_seq_len = current_len
        
        k = all_k[:current_len].reshape(compute_batch, kv_seq_len, head, dim)
        v = all_v[:current_len].reshape(compute_batch, kv_seq_len, head, dim)

        xq = q[i].view(compute_batch, 1, head, dim).transpose(1, 2).transpose(0, 1)   # shape: head, batch, 1, dim
        bmm_xq = xq.reshape(head * compute_batch, 1, dim)
        bmm_xk = k.transpose(1, 2).transpose(0, 1).transpose(2, 3).reshape(head * compute_batch, dim, kv_seq_len)
        

        # q @ k
        out = torch.bmm(bmm_xq, bmm_xk) / math.sqrt(dim)
        out = out.reshape(head, compute_batch, 1, -1).reshape(head, compute_batch, -1)

        # softmax
        out = out.softmax(-1).reshape(head, compute_batch, 1, kv_seq_len).transpose(0, 1) # shape: batch head 1 seq_len
        xv = v.transpose(1, 2) # shape: batch head, seq_len, dim
        out = torch.bmm(out.reshape(compute_batch * head, 1, kv_seq_len), xv.reshape(compute_batch * head, kv_seq_len, dim))
        
        out = out.reshape(compute_batch, head, 1, dim).view(compute_batch, head, dim)
        res.append(out)
    res = torch.cat(res)
    return res

@torch._custom_op.impl.custom_op('lightllm::copy_with_offset')
def copy_with_offset(x: Tensor, src: Tensor, start_dim: int, end_dim: int) -> Tensor:
    ...

@copy_with_offset.impl_abstract()
def lightllm_copy_with_offset_abstract(x: Tensor, src: Tensor, start_dim: int, end_dim: int) -> Tensor:
    return x

@copy_with_offset.impl(['cpu', 'cuda'])
def lightllm_copy_with_offset_impl(x, src, start_dim, end_dim) -> Tensor:
    x[start_dim:end_dim] = src
    return x
