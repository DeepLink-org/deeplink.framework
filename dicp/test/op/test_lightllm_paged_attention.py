import pytest

from dicp.vendor.AscendGraph import ext_ops
from ..common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    get_device,
    Size,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, q, all_k, all_v, q_head_num, dim, kv_head_num, block_table, seq_lengths, block_size):
        res = torch.ops.lightllm.paged_attention_inference.default(q, all_k, all_v, q_head_num, dim, kv_head_num, block_table, seq_lengths, block_size)
        return res


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestLightllmPagedAttention():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((10,), (8, 16), (8, 16)), ((10,), (8, 16), (8, 16))), Size(((10,), (16, 32), (2, 32)), ((10,), (16, 32), (2, 32)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_lightllm_paged_attention(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        
        q = torch.randn((1,) + size[1], dtype=dtype)
        k = torch.randn(size[0] + size[2], dtype=dtype)
        v = torch.randn(size[0] + size[2], dtype=dtype)

        q_head_num = size[1][0]
        dim = size[1][1]
        kv_head_num = size[2][0]
        block_table = torch.tensor([[0]], dtype=torch.int32)
        seq_lengths = list(size[0])
        block_size = 128

        dicp_q = q.to(device)
        dicp_k = k.to(device)
        dicp_v = v.to(device)
        dicp_block_table = block_table.to(device)
        dicp_seq_lengths = torch.tensor([seq_lengths], device=device, dtype=torch.int64)

        if q_head_num != kv_head_num:
            repeat = q_head_num / kv_head_num
            k = k.repeat(1, repeat, 1)
            v = v.repeat(1, repeat, 1)

        output = model(q, k, v, q_head_num, dim, kv_head_num, block_table, seq_lengths, block_size).half().reshape(-1, q_head_num, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_q, dicp_k, dicp_v, q_head_num, dim, kv_head_num, dicp_block_table, dicp_seq_lengths, block_size).reshape(-1, q_head_num, dim)

        assert torch.allclose(output, dicp_output.cpu(), rtol=1e-02, atol=1e-02, equal_nan=True)
