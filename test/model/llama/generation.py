from typing import List

import torch
from model.llama.tokenizer import Tokenizer
from model.llama.model import Transformer, WORLD_SIZE, precompute_freqs_cis


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
        device: str = "cpu"
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        max_prompt_size = params.max_prompt_size
        max_seq_len = params.max_seq_len
        assert max_prompt_size < max_seq_len, "max_prompt_size isn't smaller than max_seq_len"
        token_left_pad_id = self.tokenizer.eos_id

        total_len = min(max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(device).long()
        left_pad_size_list = []
        for k, t in enumerate(prompt_tokens):
            assert len(t) <= max_prompt_size, \
                f"prompt size of {prompts[k]}({len(t)}) is greater than max_prompt_size: {max_prompt_size}"
            left_pad_size = max_prompt_size - len(t)
            left_pad_size_list.append(left_pad_size)
            tokens[k, left_pad_size: max_prompt_size] = torch.tensor(t).to(device).long()
            if left_pad_size > 0:
                tokens[k, 0: left_pad_size] = torch.full((1, left_pad_size), token_left_pad_id).to(device).long()

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = max_prompt_size
        prev_pos = 0

        full_freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads, max_seq_len * 2
        ).to(device)
        cache_slice_size_list = [32, total_len]

        slice_size_idx = 0
        max_cur_cache_len = max([len(t) for t in prompt_tokens])

        for cur_pos in range(start_pos, total_len):
            seqlen = cur_pos - prev_pos
            if max_cur_cache_len > cache_slice_size_list[slice_size_idx]:
                slice_size_idx += 1
            slice_size = cache_slice_size_list[slice_size_idx]
            if seqlen > 1:
                # seqlen equals to max_prompt_size
                origin_mask_full = torch.zeros((1, 1, max_prompt_size, slice_size),
                                               dtype=torch.float16, device=tokens.device)
                mask_list = []
                for pad_size in left_pad_size_list:
                    origin_prompt_size = max_prompt_size - pad_size
                    right_corner_mask = torch.full((1, 1, origin_prompt_size, origin_prompt_size),
                                                   float("-inf"), device=tokens.device)
                    right_corner_mask = torch.triu(right_corner_mask, diagonal=prev_pos + 1).to(torch.float16)
                    final_mask_full = origin_mask_full.clone()
                    left_corner_mask = torch.full((1, 1, origin_prompt_size, slice_size - origin_prompt_size), float("-inf"), device=tokens.device).to(torch.float16)
                    final_mask_full[:, :, -origin_prompt_size:, -origin_prompt_size:] = right_corner_mask
                    final_mask_full[:, :, -origin_prompt_size:, :-origin_prompt_size] = left_corner_mask
                    mask_list.append(final_mask_full)
                mask = torch.cat(mask_list, dim=0)
            else:
                origin_mask_full = torch.zeros((1, 1, seqlen, slice_size),
                                               dtype=torch.float16, device=tokens.device)
                mask_list = []
                for pad_size in left_pad_size_list:
                    origin_prompt_size = max_prompt_size - pad_size
                    real_cache_size = origin_prompt_size + cur_pos - start_pos
                     # add a if condition due to tops dipu can't fill tensor with shape (1, 1, 1, 0)
                    if slice_size - real_cache_size > 0:
                        left_corner_mask = torch.full((1, 1, seqlen, slice_size - real_cache_size),
                                                       float("-inf"), device=tokens.device).to(torch.float16)
                        final_mask_full = origin_mask_full.clone()
                        final_mask_full[:, :, :, :-real_cache_size] = left_corner_mask
                    else:
                        final_mask_full = origin_mask_full.clone()
                    mask_list.append(final_mask_full)
                mask = torch.cat(mask_list, dim=0)
            mask = mask.to(device)

            logits = self.model.forward(tokens[:, prev_pos:cur_pos].clone(),
                                        torch.view_as_real(full_freqs_cis[prev_pos: cur_pos]).clone(),
                                        mask, slice_size)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
                
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            max_cur_cache_len += 1

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[left_pad_size_list[i]: max_prompt_size + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
    