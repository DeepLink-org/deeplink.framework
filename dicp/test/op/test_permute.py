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
    def forward(self, a, dims):
        res_default = torch.ops.aten.permute.default(a, dims)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestPermute():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((5, 3), (0, 1)), ((5, 3), (0, 1))),
                                       Size(((3, 5, 7), (0, 2, 1)), ((5, 3), (0, 1))),
                                       Size(((2, 3, 4, 8), (3, 0, 2, 1)), ((2, 4), (0, 1)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_permute(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        dims = size[1]

        dicp_input1 = input1.to(device)

        output = model(input1, dims)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dims)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
