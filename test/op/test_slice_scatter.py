from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, input, slice):
        res_default = torch.ops.aten.slice_scatter(input, slice, start=2, end=16, step=1)
        return  res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSliceScatter():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((16,), (14,)), ((16, 32), (3, 32))),
                                       Size(((32, 16), (14, 16)), ((32, 16), (3, 16))),
                                       Size(((32, 64, 16), (14, 64, 16)), ((32, 64), (3, 64)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_slice_scatter(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input = torch.randn(size[0], dtype=dtype)
        slice = torch.randn(size[1], dtype=dtype)

        dicp_input = input.to(device)
        dicp_slice = slice.to(device)

        output = model(input, slice)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input, dicp_slice)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
