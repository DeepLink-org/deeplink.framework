from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, input, dim, index):
        res_int = torch.ops.aten.select.int(input, dim, index)
        return res_int

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSelect():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5, 3), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("index", [-1, 1])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_select(self, sizes, dim, index, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1, dim, index)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dim, index)
        print(output)
        print(dicp_output)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
