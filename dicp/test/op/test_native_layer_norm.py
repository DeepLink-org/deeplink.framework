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
    def forward(self, x, normalized_shape, weight, bias, eps):
        res_default = torch.ops.aten.native_layer_norm.default(x, normalized_shape, weight, bias, eps)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestNativeLayerNorm():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((2, 4096, 320), (320,), (320,)),
                                            ((2, 4096, 320), (320,), (320,))),
                                       Size(((2, 1024, 640), (640,), (640)),
                                            ((2, 1024, 640), (640,), (640)))])
    @pytest.mark.parametrize("eps", [1e-05])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_native_layer_norm(self, sizes, eps, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        normalized_shape = size[1]
        input2 = torch.randn(size[2], dtype=dtype)
        input3 = torch.randn(size[2], dtype=dtype)

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)
        dicp_input3 = input3.to(device)

        output = model(input1, normalized_shape, input2, input3, eps)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, normalized_shape, dicp_input2, dicp_input3, eps)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), atol=1e-06, equal_nan=True)
