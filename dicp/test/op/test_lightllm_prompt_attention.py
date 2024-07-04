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
    def forward(self, q, k, v, seqlen, num_head, head_dim, num_key_value_heads):
        res = torch.ops.lightllm.prompt_attention_inference.default(q, k, v, seqlen, num_head, head_dim, num_key_value_heads)
        return res


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestLightllmPromptAttention():
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("sizes", [Size(((1, 32, 16, 32), (32,)), ((1, 32, 16, 32), (32,))), Size(((1, 32, 16, 64), (32,)), ((1, 32, 16, 64), (32,)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_lightllm_prompt_attention(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        input2 = torch.randn(size[0], dtype=dtype)
        input3 = torch.randn(size[0], dtype=dtype)
        input4 = torch.tensor(size[1], dtype=torch.int32)
        num_head = size[0][2]
        head_dim = size[0][3]
        num_key_value_heads = size[0][2]

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)
        dicp_input3 = input3.to(device)
        dicp_input4 = input4.to(device)

        output = model(input1, input2, input3, input4, num_head, head_dim, num_key_value_heads).view(size[1][0], num_head * head_dim).half()
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1.view(1, -1, num_head * head_dim), dicp_input2.view(1, -1, num_head * head_dim), dicp_input3.view(1, -1, num_head * head_dim), dicp_input4, num_head, head_dim, num_key_value_heads).view(size[1][0], num_head * head_dim)

        assert torch.allclose(output, dicp_output.cpu(), rtol=1e-02, atol=1e-02, equal_nan=True)
