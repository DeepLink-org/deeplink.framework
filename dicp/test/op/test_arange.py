import pytest
from ..common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, a, b):
        res_default = torch.ops.aten.arange.default(b)
        res_start = torch.ops.aten.arange.start(a, b)
        return res_default, res_start


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestArange():
    @pytest.mark.parametrize("start", [0, 1, 2])
    @pytest.mark.parametrize("end", [5, 7, 9])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_arange(self, start, end, compiled_model):
        output = model(start, end)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(start, end)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), equal_nan=True)
