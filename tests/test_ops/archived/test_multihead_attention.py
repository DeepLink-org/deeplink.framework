import torch
import torch_dipu
import torch.nn as nn


embed_dim = 512
num_heads = 64

multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

batch_size = 2
sequence_len = 1024
query = torch.randn(sequence_len, batch_size, embed_dim)
key = torch.randn(sequence_len, batch_size, embed_dim)
value = torch.randn(sequence_len, batch_size, embed_dim)

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
attn_output_d, attn_output_weights_d = multihead_attn.cuda()(query_d, key_d, value_d)
attn_output_d.backward(torch.ones_like(attn_output_d))

assert torch.allclose(attn_output, attn_output_d.cpu(), atol = 1e-3)
assert torch.allclose(attn_output_weights, attn_output_weights_d.cpu(), atol = 1e-3)

assert torch.allclose(query.grad, query_d.grad.cpu(), atol = 1e-3)
assert torch.allclose(key.grad, key_d.grad.cpu(), atol = 1e-3)
assert torch.allclose(value.grad, value_d.grad.cpu(), atol = 1e-3)