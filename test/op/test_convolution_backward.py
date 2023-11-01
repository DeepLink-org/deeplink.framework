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
    def forward(self, grad_output, x, weight):
        res_default = torch.ops.aten.convolution_backward.default(grad_output, x, weight, bias_sizes=[0],
                                                                  stride=[1, 1], padding=[0, 0], dilation=[1, 1],
                                                                  transposed=False, output_padding=[0, 0],
                                                                  groups=1, output_mask=[True, True, False])
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestConvolutionBackward():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((32, 2048, 7, 7), (32, 512, 7, 7), (2048, 512, 1, 1)),
                                            ((32, 2048, 7, 7), (32, 512, 7, 7), (2048, 512, 1, 1)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_convolution_backward(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        grad_output = torch.randn(size[0], dtype=dtype, requires_grad=True)
        inputs = torch.randn(size[1], dtype=dtype)
        weight = torch.randn(size[2], dtype=dtype, requires_grad=True)

        dicp_grad_output = grad_output.to(device)
        dicp_inputs = inputs.to(device)
        dicp_weight = weight.to(device)

        output = model(grad_output, inputs, weight)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_grad_output, dicp_inputs, dicp_weight)

        for i, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                assert torch.allclose(item.detach(), dicp_output[i].cpu().detach(), atol=1e-02, equal_nan=True)
            else:
                assert item == dicp_output[i]
