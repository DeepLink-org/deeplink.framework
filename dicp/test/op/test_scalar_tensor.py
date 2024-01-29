import pytest
from ..common.utils import (
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


class AscendOpModule(torch.nn.Module):
    def forward(self, a, redundant_input, dtype):
        # If there is only one operator called scalar_tensor,
        # ascend graph compiler will give an error:
        # GE.. [Check][Param] SetInputs failed: input operator size can not be 0.
        # To solve this problem, an additional redundant input is added,
        # and the result of an addition operator is returned.
        scalar_tensor = torch.ops.aten.scalar_tensor.default(a, dtype=dtype)
        res_default = torch.ops.aten.add.Tensor(redundant_input, scalar_tensor)
        return scalar_tensor, res_default


model = OpModule()
ascend_model = AscendOpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)
ascend_compiled_model = compile_model(ascend_model, args.backend, args.dynamic)


class TestScalarTensor():
    @pytest.mark.skipif(args.backend == 'ascendgraph', reason="skip ascendgraph")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float16])
    @pytest.mark.parametrize("inputs", [1.0, 3.0, 0.0])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_scalar_tensor(self, inputs, dtype, compiled_model):
        output = model(inputs, dtype)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(inputs, dtype)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)

    @pytest.mark.skipif(args.backend == 'topsgraph', reason="skip topsgraph")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float16])
    @pytest.mark.parametrize("inputs", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("compiled_model", ascend_compiled_model)
    def test_torch_ascend_scalar_tensor(self, inputs, dtype, compiled_model):
        redundant_input = torch.ones(1, dtype=dtype)
        output, _ = ascend_model(inputs, redundant_input, dtype)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output, _ = compiled_model.model(inputs, redundant_input, dtype)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
