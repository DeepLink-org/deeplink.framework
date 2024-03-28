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
    def forward(self, operand, src, dim, index):
        res_value = torch.ops.aten.select_scatter.default(operand, src, dim, index)
        return res_value


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSelectScatter():
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("sizes", [Size(((16, 2, 32), (16, 32), 1, 0), ((16, 2, 32), (16, 32), 1, 0)),
                                       Size(((3, 8, 16), (8, 16), 0, 2), ((3, 8, 16), (8, 16), 0, 2))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_select_scatter(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        input2 = torch.randn(size[1], dtype=dtype)
        dim = size[2]
        index = size[3]

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(input1, input2, dim, index)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_input2, dim, index)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
