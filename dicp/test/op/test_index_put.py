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
    def forward(self, x, indices, values):
        res_Tensor = torch.ops.aten.index_put.default(x, indices, values)
        return res_Tensor


model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)


class TestIndexPut():
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("sizes", [Size(((1, 32, 208, 128), (None, None, (6,)), (32, 6, 128)),
                                            ((1, 32, 208, 128), (None, None, (6,)), (32, 6, 128))),
                                       Size(((1, 2, 10, 8, 7, 11), (None, None, (2, 3), (4, 1, 1), (1, 2, 1), None),
                                             (1, 1, 4, 1, 3, 11)),
                                            ((1, 2, 10, 8 ,7, 11), (None, None, (2, 3), (4, 1, 1), (1, 2, 1), None),
                                             (1, 1, 4, 1, 3, 11))),
                                       Size(((1, 2, 10, 8, 7, 11), (None, None, (2, 3), (4, 1, 1), None, (1, 2, 1)),
                                             (4, 2, 3, 1, 2, 7)),
                                            ((1, 2, 10, 8 ,7, 11), (None, None, (2, 3), (4, 1, 1), None, (1, 2, 1)),
                                             (4, 2, 3, 1, 2, 7)))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch_split(self, sizes, dtype, compiled_model):
        device = get_device()
        size = sizes.dynamic if compiled_model.dynamic else sizes.static
        x_size = size[0]
        indices_size_tuple = size[1]
        values_size = size[2]

        input1 = torch.randn(x_size, dtype=dtype)
        indices = []
        for dim_idx, idx_size in enumerate(indices_size_tuple):
            if idx_size is None:
                indices.append(None)
            else:
                indices.append(torch.randint(x_size[dim_idx], idx_size, dtype=torch.int32))
        value = torch.randn(values_size, dtype=dtype)
        dicp_input1 = input1.to(device)
        dicp_indices = [None if index is None else index.to(device) for index in indices]
        dicp_value = value.to(device)

        output = model(input1, indices, value)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, dicp_indices, dicp_value)

        assert torch.allclose(output.cpu(), dicp_output.cpu(), equal_nan=True)
