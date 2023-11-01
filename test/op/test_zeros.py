import pytest
from common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    Size,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, size):
        res_default = torch.ops.aten.zeros.default(size)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestZeros():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_zeros(self, sizes, dtype, compiled_model):
        size = sizes.dynamic if compiled_model.dynamic else sizes.static

        output = model(size)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(size)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
