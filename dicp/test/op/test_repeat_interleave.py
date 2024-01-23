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
    def forward(self, x, repeats, dim):
        res_default = x.repeat_interleave(repeats, dim)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestRepeat():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((3, 5), 5, 0), ((3, 5), 5, 0)),
                                       Size(((4, 6, 8), 2, 1), ((4, 6, 8), 2, 1)),
                                       Size(((4, 3, 2, 2), 3, -1), ((4, 3, 2, 2), 3, -1))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_repeat_self_int(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size[0], dtype=dtype)
        repeats = size[1]
        dim = size[2]

        dicp_input1 = input1.to(device)

        output = model(input1, repeats, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, repeats, dim)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
