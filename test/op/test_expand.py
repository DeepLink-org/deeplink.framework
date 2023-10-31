from common.utils import *


class OpModule(torch.nn.Module):
    def forward(self, a, b):
        res_default = torch.ops.aten.expand.default(a, b)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestExpand():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((3, 1), (3, 4)), ((3, 1), (3, 5))),
                                       Size(((5, 3, 1), (5, 3, 4)), ((5, 1), (5, 3)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_expand(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        expand_size = size[1]

        dicp_input1 = input1.to(device)

        output = model(input1, expand_size)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, expand_size)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
