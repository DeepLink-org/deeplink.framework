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
    def forward(self, a, b):
        res_default = torch.ops.aten.view(a, b)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestView():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((5,), (5, 1)), ((5, 3), (3, 5))),
                                       Size(((5, 3), (3, 5)), ((5, 3), (3, 5))),
                                       Size(((5, 3), (-1, 5)), ((5, 3), (-1, 5))),
                                       Size(((5, 3), (3, -1)), ((5, 3), (3, -1))),
                                       Size(((2, 8), (16,)), ((2, 8), (16,))),
                                       Size(((2, 8), (2, 4, 2)), ((2, 8), (2, 4, 2)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_view(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        view_size = size[1]

        dicp_input1 = input1.to(device)

        output = model(input1, view_size)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, view_size)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
