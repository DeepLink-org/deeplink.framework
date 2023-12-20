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
        res_correction = torch.ops.aten.var_mean.correction(a, b, correction=0, keepdim=True)
        return res_correction


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestVarMean():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("dim", [[0], [1], [0, -1]])
    @pytest.mark.parametrize("sizes", [Size((77, 1027), (77, 1024)),
                                       Size((4, 77, 1024), (4, 1024)),
                                       Size((2, 32, 10, 9216), (4, 77))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_var_mean(self, sizes, dtype, dim, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dim)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), atol=1e-3, equal_nan=True)
