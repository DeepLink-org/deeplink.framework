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
    def forward(self, x, split_size_or_sections, dim):
        res_Tensor = torch.ops.aten.split.Tensor(x, split_size_or_sections, dim)
        for i in range(len(res_Tensor)):
            res_Tensor[i] = res_Tensor[i] + 1.0
        return res_Tensor


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestSplit():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5, 3), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("split_size_or_sections", [1, 2])
    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_split(self, sizes, split_size_or_sections, dim, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device)

        output = model(input1, split_size_or_sections, dim)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, split_size_or_sections, dim)

        for i, item in enumerate(output):
            assert torch.allclose(item, dicp_output[i].cpu(), equal_nan=True)
