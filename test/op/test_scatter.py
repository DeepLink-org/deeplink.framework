from common.utils import *


class OpModule(torch.nn.Module):
    def forward(self, a, b):
        res_value = torch.ops.aten.scatter.value(a, 0, b, 5.0)
        return res_value


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestScatter():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((5,), (0, 2)), ((5, 3), ((0, 2), (0, 2)))),
                                       Size(((3, 5), ((0, 2), (0, 2))), ((5, 3), ((0, 2), (0, 2))))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_scatter(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        input2 = torch.tensor(size[1])

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(input1, input2)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_input2)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
