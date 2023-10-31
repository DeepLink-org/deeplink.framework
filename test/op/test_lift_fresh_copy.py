from common.utils import *


class OpModule(torch.nn.Module):
    def forward(self, a, b, c):
        res_default1 = torch.tensor(torch.finfo(a.dtype).min)
        res_default2 = torch.tensor([b, c])
        return res_default1, res_default2


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestLiftFreshCopy():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_lift_fresh_copy(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)
        value1 = round(random.random(), 4)
        value2 = round(random.random(), 4)

        dicp_input1 = input1.to(device)

        output = model(input1, value1, value2)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, value1, value2)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), equal_nan=True)
