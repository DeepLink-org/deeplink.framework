# Copyright (c) 2023, DeepLink.
import torch
import torch.nn as nn
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestMultiheadAttention(TestCase):
    def test_multihead_attention(self):
        EMBED_DIM = 512
        NUM_HEADS = 64

        multihead_attn = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS)

        BATCH_SIZE = 2
        SEQUENCE_LEN = 1024
        query = torch.randn(SEQUENCE_LEN, BATCH_SIZE, EMBED_DIM)
        key = torch.randn(SEQUENCE_LEN, BATCH_SIZE, EMBED_DIM)
        value = torch.randn(SEQUENCE_LEN, BATCH_SIZE, EMBED_DIM)

        query_d = query.cuda()
        key_d = key.cuda()
        value_d = value.cuda()

        query.requires_grad = True
        value.requires_grad = True
        key.requires_grad = True

        attn_output, attn_output_weights = multihead_attn(query, key, value)
        attn_output.backward(torch.ones_like(attn_output))

        query_d.requires_grad = True
        value_d.requires_grad = True
        key_d.requires_grad = True
        attn_output_d, attn_output_weights_d = multihead_attn.cuda()(
            query_d, key_d, value_d
        )
        attn_output_d.backward(torch.ones_like(attn_output_d))

        self.assertEqual(attn_output, attn_output_d.cpu(), prec=1e-3)
        self.assertEqual(attn_output_weights, attn_output_weights_d.cpu(), prec=1e-3)

        self.assertEqual(query.grad, query_d.grad.cpu(), prec=1e-3)
        self.assertEqual(key.grad, key_d.grad.cpu(), prec=1e-3)
        self.assertEqual(value.grad, value_d.grad.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
