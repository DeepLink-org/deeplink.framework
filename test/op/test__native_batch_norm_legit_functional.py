from common.utils import *


class OpModule(torch.nn.Module):
    def forward(self, a, device="cpu"):
        m = torch.nn.BatchNorm2d(100)
        m.to(device)
        res_default = m(a)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestNativeBatchNormLegitFunctional():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((20, 100, 35, 45), (20, 100, 35, 45)),
                                       Size((30, 100, 45, 35), (30, 100, 45, 35))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch__native_batch_norm_legit_functional(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, device)

        assert torch.allclose(output.detach(), dicp_output.cpu().detach(), equal_nan=True)
