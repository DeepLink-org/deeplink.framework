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
    def forward(self, x, slice, dim):
        res_default = torch.ops.aten.slice_scatter.default(x, slice, dim=dim, start=2, end=16, step=1)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSliceScatter():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((16,), (14,), 0), ((16, 32), (3, 32), 0)),
                                       Size(((32, 16), (32, 14), 1), ((32, 16), (32, 14), 1)),
                                       Size(((32, 64, 16), (32, 64, 14), 2), ((32, 64), (33, 62), 2))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_slice_scatter(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        input2 = torch.randn(size[1], dtype=dtype)
        dim = size[2]

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(input1, input2, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_input2, dim)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
