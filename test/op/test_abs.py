import torch
import op.utils as op_utils
import torch_dipu
import pytest
import torch._dynamo as dynamo

class OpModule(torch.nn.Module):
    def forward(self, a):
        res = torch.abs(a)
        return res

model = OpModule()
args = op_utils.parse_args()
compiled_models = op_utils.compile_model(model, args.backend, args.need_dynamic)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("size", [(5, 3), (3, 5), (2, 4)])
@pytest.mark.parametrize("compiled_model", compiled_models)
def test_torch_abs_default(size, dtype, compiled_model):
    device = op_utils.get_device()
    input1 = torch.randn(size, dtype=dtype)

    dicp_input1 = input1.to(device)

    output = model(input1)
    dynamo.reset()
    op_utils.update_dynamo_config(compiled_model.dynamic)
    dicp_output = compiled_model.model(dicp_input1)

    assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
