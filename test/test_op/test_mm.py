import torch
import torch_dipu
import pytest
import utils
import torch._dynamo as dynamo

class OpModule(torch.nn.Module):
    def forward(self, a, b):
        res = torch.mm(a, b)
        return res

model = OpModule()
args = utils.parse_args()
compiled_models = utils.compile_model(model, args.backend, args.need_dynamic)

class Size():
    def __init__(self, size1, size2) -> None:
        self.size1 = size1
        self.size2 = size2


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("size", [Size((5, 3), (3, 5)), Size((3, 5), (5, 3))])
@pytest.mark.parametrize("compiled_model", compiled_models)
def test_torch_mm_default(size, dtype, compiled_model):
    device = utils.get_device()
    input1 = torch.randn(size.size1, dtype=dtype)
    input2 = torch.randn(size.size2, dtype=dtype)

    dicp_input1 = input1.to(device)
    dicp_input2 = input2.to(device)

    output = model(input1, input2)
    dynamo.reset()
    utils.update_dynamo_config(compiled_model.dynamic)
    dicp_output = compiled_model.model(dicp_input1, dicp_input2)

    assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
