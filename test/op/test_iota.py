from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, length, start, step, device="cpu"):
        res_default = torch.ops.prims.iota.default(length, start=start, step=step, dtype=torch.int64, device=device, requires_grad=False)
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestIota():
    @pytest.mark.parametrize("length", [8, 16])
    @pytest.mark.parametrize("start", [0, 2])
    @pytest.mark.parametrize("step", [5, 9])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_iota(self, length, start, step, compiled_model):
        output = model(length, start, step)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(length, start, step)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
