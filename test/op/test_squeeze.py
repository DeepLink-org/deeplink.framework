from common.utils import *


class OpModule(torch.nn.Module):
    def forward(self, a, dim):
        res_dim = torch.ops.aten.squeeze.dim(a, dim=dim)
        return res_dim


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSqueeze():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5, 1), (5, 1)),
                                       Size((3, 1, 4), (5, 1)),
                                       Size((2, 1, 3, 4), (2, 1))])
    @pytest.mark.parametrize("dim", [1])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_squeeze(self, sizes, dim, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)
        dim = -1 if len(size) == 2 else dim

        dicp_input1 = input1.to(device)

        output = model(input1, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dim)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
