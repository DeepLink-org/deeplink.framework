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
    def forward(self, x, weight, bias=None):
        res_default = torch.ops.aten.convolution.default(x, weight, bias=bias, stride=[2, 2],
                                                         padding=[3, 3], dilation=[1, 1],
                                                         transposed=False, output_padding=[0, 0],
                                                         groups=1)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestConvolution():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((32, 3, 224, 224), (64, 3, 7, 7)), ((32, 3, 224, 224), (64, 3, 7, 7)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_convolution(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        inputs = torch.randn(size[0], dtype=dtype)
        weights = torch.randn(size[1], dtype=dtype)

        dicp_inputs = inputs.to(device)
        dicp_weights = weights.to(device)

        output = model(inputs, weights)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_inputs, dicp_weights)

        assert torch.allclose(output.detach(), dicp_output.cpu().detach(), atol=1e-02, equal_nan=True)
