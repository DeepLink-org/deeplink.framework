from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, a, size, stride):
        res_default = torch.ops.aten.new_empty_strided.default(a, size, stride)
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestNewEmptyStrided():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_new_empty_strided(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)
        size = input1.size()
        stride = input1.stride()

        dicp_input1 = input1.to(device)
        dicp_size = dicp_input1.size()
        dicp_stride = dicp_input1.stride()

        output = model(input1, size, stride)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_size, dicp_stride)

        assert output.size() == dicp_output.size()
