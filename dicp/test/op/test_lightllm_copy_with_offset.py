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
    def forward(self, out, k, start_dim, end_dim):
        res = torch.ops.lightllm.copy_with_offset.default(out, k, start_dim, end_dim)
        return res


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestLightllmCopyWithOffset():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((8, 8, 16), (6, 8, 16)), ((8, 8, 16), (6, 8, 16))), Size(((8, 16, 32), (6, 16, 32)), ((8, 16, 32), (6, 16, 32)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_lighllm_copy_with_offset(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        input2 = torch.randn(size[1], dtype=dtype)
        start_dim = 0
        end_dim = 6

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(input1, input2, start_dim, end_dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_input2, start_dim, end_dim)

        assert torch.allclose(output, dicp_output.cpu(), rtol=1e-02, atol=1e-02, equal_nan=True)
