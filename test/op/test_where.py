from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, condition, a, b):
        res_self1 = torch.ops.aten.where(condition, a, b)
        res_self2 = torch.ops.aten.where(condition, a, b)
        return res_self1, res_self2

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestWhere():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_where(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        condition = torch.randint(2, size) == torch.randint(2, size)
        input1 = torch.randn(size, dtype=dtype)
        input2 = torch.randn(size, dtype=dtype)

        dicp_condition = condition.to(device)
        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(condition, input1, input2)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_condition, dicp_input1, dicp_input2)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), equal_nan=True)
