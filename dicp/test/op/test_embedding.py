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
    def forward(self, weight, indices):
        res_default = torch.ops.aten.embedding.default(weight, indices)
        return res_default


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestEmbedding():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size(((100, 32), (100, (1, 10))), ((100, 32), (100, (1, 10)))),
                                       Size(((1000, 4096), (1000, (1, 12))), ((1000, 4096), (1000, (1, 12))))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_embedding(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        weight = torch.randn(size[0], dtype=dtype)
        indices = torch.randint(*size[1])

        dicp_weight = weight.to(device)
        dicp_indices = indices.to(device)

        output = model(weight, indices)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_weight, dicp_indices)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True)
