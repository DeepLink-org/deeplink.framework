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
    def forward(self, x, groups, weight, bias):
        res_default = torch.ops.aten.group_norm.default(x, groups, weight, bias)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestGroupNorm():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((2, 320, 64, 64), (320,), (320,)),
                                            ((2, 320, 64, 64), (320,), (320,))),
                                       Size(((2, 640, 32, 32), (640,), (640)),
                                            ((2, 640, 32, 32), (640,), (640)))])
    @pytest.mark.parametrize("groups", [32])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_group_norm(self, sizes, groups, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        input2 = torch.randn(size[1], dtype=dtype)
        input3 = torch.randn(size[2], dtype=dtype)

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)
        dicp_input3 = input3.to(device)

        output = model(input1, groups, input2, input3)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, groups, dicp_input2, dicp_input3)

        assert torch.allclose(output, dicp_output.cpu(), atol=1e-06, equal_nan=True)
