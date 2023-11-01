import pytest
from common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, a, dtype):
        res_default = torch.ops.aten.scalar_tensor.default(a, dtype=dtype)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestScalarTensor():
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float16])
    @pytest.mark.parametrize("inputs", [1.0, 3.0, 0.0])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_scalar_tensor(self, inputs, dtype, compiled_model):
        output = model(inputs, dtype)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(inputs, dtype)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
