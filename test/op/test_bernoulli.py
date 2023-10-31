from common.utils import *


class OpModule(torch.nn.Module):
    def forward(self, x, p):
        res_p = torch.ops.aten.bernoulli.p(x, p)
        return res_p


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestBernoulli():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_bernoulli(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)
        p = round(random.random(), 4)

        dicp_input1 = input1.to(device)

        output = model(input1, p)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, p)

        assert output.size() == dicp_output.size()
