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
    def forward(self, a, diagonal):
        res = torch.ops.aten.tril.default(a, diagonal=diagonal)
        return res


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestTril():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("diagonal", [0, 1])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_transpose(self, sizes, dtype, diagonal, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype) if isinstance(size, tuple) else torch.tensor(size, dtype=dtype)
        diagonal = diagonal if isinstance(diagonal, tuple) else min(diagonal, 0)

        dicp_input1 = input1.to(device)

        output = model(input1, diagonal)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, diagonal)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
