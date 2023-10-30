from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, input, dim, index):
        res_default = torch.ops.aten.gather(input, dim, index)
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestGather():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5, 3), (5, 3)), Size((3, 5), (5, 3)), Size((2, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_gather(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input = torch.randn(size, dtype=dtype)
        dim = 1
        index = torch.tensor([[0, 0], [1, 0]])

        dicp_input = input.to(device)
        dicp_index = index.to(device)

        output = model(input, dim, index)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input, dim, dicp_index)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
