from typing import Optional, Tuple, List
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from common.utils import get_device
device = get_device()


WORLD_SIZE = 1


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    max_prompt_size: int = 32


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    # assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_real: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = torch.view_as_complex(freqs_cis_real)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // WORLD_SIZE
        self.head_dim = args.dim // args.n_heads
        self.init_slice_size = args.max_seq_len - args.max_prompt_size

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.cache_k = torch.empty(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).to(device)
        self.cache_v = torch.empty(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).to(device)

    def forward(self, x: torch.Tensor, freqs_cis_real: torch.Tensor, mask: torch.Tensor, slice_size: int):
        # x: bsz, seqlen, params, dim
        # if seqlen > 1
        # cache_k: bsz, max_seq_len - max_prompt_size, n_local_heads, head_dim
        # cache_v: bsz, max_seq_len - max_prompt_size, n_local_heads, head_dim
        # else seqlen = 1
        # cache_k: bsz, max_seq_len, n_local_heads, head_dim
        # cache_v: bsz, max_seq_len, n_local_heads, head_dim
        # freqs_cis_real: seqlen, head_dim // 2, 2
        # mask: bsz, 1, seqlen, max_seq_len
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis_real=freqs_cis_real)

        # TODO: for now(20230612) we can't handle dynamic batch_size
        # if we want handle dynamic batch_size self.cache_k and self.cache_v
        # must use torch.slice_scatter to update data
        # e.g. self.cache_k[:bsz] = torch.cat([self.cache_k[:bsz, 1:], xk], dim=1)
        # we should implement torch.ops.aten.slice_scatter.default
        if seqlen > 1:
            # when seqlen > 1, seqlen equals to max_prompt_size
            self.cache_k = torch.cat([self.cache_k[:bsz, :self.init_slice_size], xk], dim=1)
            self.cache_v = torch.cat([self.cache_v[:bsz, :self.init_slice_size], xv], dim=1)
        else:
            self.cache_k = torch.cat([self.cache_k[:bsz, 1:], xk], dim=1)
            self.cache_v = torch.cat([self.cache_v[:bsz, 1:], xv], dim=1)

        xq = xq.transpose(1, 2)
        # xq: bsz, n_local_heads, seqlen, head_dim
        keys = self.cache_k[:bsz, -slice_size:].transpose(1, 2)
        # keys: bsz, n_local_heads, slice_size, head_dim
        values = self.cache_v[:bsz, -slice_size:].transpose(1, 2)
        # values: bsz, n_local_heads, slice_size, head_dim
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores: bsz, n_local_heads, seqlen, slice_size
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        # output: bzs, n_local_heads, seqlen, head_dim
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis_real: torch.Tensor, mask: torch.Tensor, slice_size: int):
        # x: bsz, seqlen, params, dim
        # cache_k: bsz, history_seqlen, self.n_heads / WORLD_SIZE, self.head_dim
        # cache_v: bsz, history_seqlen, self.n_heads / WORLD_SIZE, self.head_dim
        # freqs_cis_real: seqlen, self.head_dim // 2, 2
        # mask: 1, 1, seqlen, seqlen + history_seqlen
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis_real, mask, slice_size)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # self.freqs_cis = precompute_freqs_cis(
        #     self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        # )

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, freqs_cis_real: torch.Tensor, mask: torch.Tensor, slice_size: int):
        # _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # h: bsz, seqlen, params, dim
        # self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        for layer_idx in range(len(self.layers)):
        # for layer_idx in range(1):
            h = self.layers[layer_idx](h, freqs_cis_real, mask, slice_size)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()