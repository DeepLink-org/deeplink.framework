from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, a, b):
        res_Tensor = torch.ops.aten.mul.Tensor(a, b)
        res_Scalar = torch.ops.aten.mul.Scalar(res_Tensor, 2.0)
        return res_Scalar

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestMul():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_mul(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)
        input2 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)
        dicp_input2 = input2.to(device)

        output = model(input1, input2)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_input2)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
