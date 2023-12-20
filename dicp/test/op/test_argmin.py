import pytest
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
    def forward(self, a, dim, keepdim):
        res_default = torch.ops.aten.argmin.default(a, dim, keepdim)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestArgmin():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("dim", [-1, 0])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_argmin(self, sizes, dim, keepdim, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1, dim, keepdim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dim, keepdim)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
