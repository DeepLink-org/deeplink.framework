from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, a, b):
        res_default = torch.ops.aten.gelu.default(a, approximate=b)
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestGelu():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_gelu(self, sizes, approximate, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1, approximate)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, approximate)

        assert torch.allclose(output, dicp_output.cpu(), atol=1e-02, equal_nan=True)
