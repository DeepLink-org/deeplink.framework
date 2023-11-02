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
    def forward(self, grad_output, x):
        res_default, indices = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])
        res_default = torch.ops.aten.max_pool2d_with_indices_backward(grad_output, x, kernel_size=[3, 3],
                                                                      stride=[2, 2], padding=[1, 1], dilation=[1, 1],
                                                                      ceil_mode=False, indices=indices)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestMaxPool2dWithIndicesBackward():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((32, 64, 56, 56), (32, 64, 112, 112)), ((32, 64, 56, 56), (32, 64, 112, 112)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_max_pool2d_with_indices_backward(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        grad_output = torch.randn(size[0], dtype=dtype)
        inputs = torch.randn(size[1], dtype=dtype)

        dicp_grad_output = grad_output.to(device)
        dicp_inputs = inputs.to(device)

        output = model(grad_output, inputs)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_grad_output, dicp_inputs)

        assert torch.allclose(output.detach(), dicp_output.cpu().detach(), atol=1e-02, equal_nan=True)
