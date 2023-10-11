from common.utils import *

class OpModule(torch.nn.Module):
    def forward(self, inputs, weights, outputs, device="cpu"):
        m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(0, 0), dilation=(1, 1), bias=False)
        m.to(device)
        m.weights = weights
        res_default = m(inputs)
        loss = torch.nn.CrossEntropyLoss()
        res_loss = loss(res_default, outputs)
        res_loss.backward()
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestConvolutionBackward():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((20, 16, 50, 100), (33, 16, 3, 5), (20, 33, 24, 96)), ((20, 16, 50, 100), (33, 16, 3, 5), (20, 33, 24, 96)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_convolution_backward(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        inputs = torch.randn(size[0], dtype=dtype)
        weights = torch.randn(size[1], dtype=dtype)
        outputs = torch.randn(size[2], dtype=dtype, requires_grad=True)

        dicp_inputs = inputs.to(device)
        dicp_weights = weights.to(device)
        dicp_outputs = outputs.to(device)

        output = model(inputs, weights, outputs)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_inputs, dicp_weights, dicp_outputs, device)
        print("**************************", flush=True)
        print(inputs.size(), weights.size(), output.size(), flush=True)

        assert torch.allclose(output.detach(), dicp_output.cpu().detach(), atol=1e-02, equal_nan=True)
