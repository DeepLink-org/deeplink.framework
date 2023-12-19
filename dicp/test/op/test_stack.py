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
    def forward(self, a, b, c):
        res_default1 = torch.ops.aten.stack.default((a, b), c)
        res_default2 = torch.ops.aten.stack.default((a, b, a), c)
        res_default3 = torch.ops.aten.stack.default((a,), c)
        return res_default1, res_default2, res_default3


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestStack():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (3, 5)),
                                       Size((2, 5), (2, 5)),
                                       Size((2, 3, 4), (2, 3, 4))])
    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_stack(self, sizes, dim, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)
        input2 = torch.randn(size, dtype=dtype)
        dim = min(dim, len(size) - 1)

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(input1, input2, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_input2, dim)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), equal_nan=True)
