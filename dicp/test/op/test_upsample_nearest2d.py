import pytest
from ..common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    get_device,
    Size,
    update_dynamo_config,
)


class OpModule(torch.nn.Module):
    def forward(self, x, output_size, scales_h, scales_w):
        res_default = torch.ops.aten.upsample_nearest2d.default(x, output_size, scales_h, scales_w)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestUpsampleNearest2d():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((2, 1280, 8, 8), (16, 16)), ((2, 1280, 8, 8), (16, 16))),
                                       Size(((2, 1280, 16, 16), (32, 32)), ((2, 1280, 16, 16), (32, 32))),
                                       Size(((2, 1280, 32, 32), (64, 64)), ((2, 1280, 32, 32), (64, 64)))])
    @pytest.mark.parametrize("scales_h", [2.0])
    @pytest.mark.parametrize("scales_w", [2.0])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_upsample_nearest2d(self, sizes, scales_h, scales_w, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        output_size = size[1]

        dicp_input1 = input1.to(device)

        output = model(input1, output_size, scales_h, scales_w)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, output_size, scales_h, scales_w)

        assert torch.equal(output, dicp_output.cpu())
