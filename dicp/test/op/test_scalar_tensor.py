import pytest
from ..common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, a, redundant_input, dtype):
        # If there is only one operator called scalar_tensor,
        # ascend graph compiler will give an error:
        # GE.. [Check][Param] SetInputs failed: input operator size can not be 0.
        # To solve this problem, an additional redundant input is added,
        # and the result of an addition operator is returned.
        res_default = torch.ops.aten.scalar_tensor.default(a, dtype=dtype)
        return res_default + redundant_input, res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestScalarTensor():
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float16])
    @pytest.mark.parametrize("inputs", [1.0, 3.0, 0.0])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_scalar_tensor(self, inputs, dtype, compiled_model):
        redundant_input = torch.randn(1, dtype=dtype)
        _, output = model(inputs, redundant_input, dtype)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        _, dicp_output = compiled_model.model(inputs, redundant_input, dtype)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
