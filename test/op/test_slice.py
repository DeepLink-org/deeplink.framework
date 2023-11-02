import pytest
from common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    get_device,
    Size,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, a, dim, start, end):
        res_Tensor = torch.ops.aten.slice.Tensor(a, dim=dim, start=start, end=end)
        res_Tensor = res_Tensor + 1.0
        return res_Tensor


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSlice():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)),
                                       Size((3, 5), (5, 3)),
                                       Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.parametrize("start", [0, 1, None])
    @pytest.mark.parametrize("end", [2, None])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_slice(self, sizes, dim, start, end, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1, dim, start, end)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dim, start, end)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
