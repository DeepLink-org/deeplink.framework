from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, a):
        res_default = torch.ops.aten.scalar_tensor.default(a)
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestScalarTensor():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("input", [1.0, 3.0, 0.0])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_scalar_tensor(self, input, dtype, compiled_model):
        output = model(input)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(input)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
