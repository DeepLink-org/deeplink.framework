import pytest
import operator
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
    def forward(self, a, b):
        split = torch.ops.aten.split.Tensor(a, 1)
        res = operator.getitem(split, b)
        return res


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)

torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.assume_static_by_default = True

class TestSequenceAt():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("dim", [0])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_operator_sequence_at(self, sizes, dim, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static

        in_tensor = torch.randn(size, dtype=dtype)
        dicp_input = in_tensor.to(device)

        output = model(in_tensor, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input, dim)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
