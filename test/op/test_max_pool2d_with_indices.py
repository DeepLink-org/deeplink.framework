from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, inputs):
        res_default, indices = torch.ops.aten.max_pool2d_with_indices.default(inputs, [3, 3], [2, 2], [1, 1])
        return res_default, indices

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestMaxPool2dWithIndices():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((32, 64, 112, 112), (32, 64, 112, 112))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_max_pool2d_with_indices(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        inputs = torch.randn(size, dtype=dtype)

        dicp_inputs = inputs.to(device)

        output = model(inputs)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_inputs)

        for i, item in enumerate(output):
            if i == 1 and args.backend == "topsgraph":
                assert item.size() == dicp_output[i].size()
            else:
                assert torch.allclose(item.detach(), dicp_output[i].cpu().detach(), atol=1e-02, equal_nan=True)
            
