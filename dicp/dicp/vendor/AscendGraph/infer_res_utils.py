from collections.abc import Sequence
from typing import Optional, Tuple, Union
from dicp.dynamo_bridge.utils import get_memory_format

import torch
import math

"""parse and get val"""


# in conversion.py, some ops' ("cast") inputs are ascend_type like 'FLOAT',but infer needs torch type
def ascend_type_to_torch(ascend_type: str) -> torch.dtype:
    ascend_type_map = {
        "BOOL": torch.bool,
        "INT64": torch.int64,
        "FLOAT": torch.float32,
        "FLOAT16": torch.float16,
        "INT32": torch.int32,
        "COMPLEX64": torch.complex64,
    }

    assert (
        ascend_type in ascend_type_map
    ), "unknow ascend_dtype in ascend_type_to_torch!"

    return ascend_type_map[ascend_type]


def get_fake_tensor_meta_val(
    x, req_dim=True, req_dtype=True
) -> Tuple[torch.Tensor, Union[torch.Size, list], int, Union[torch.dtype, type, None]]:
    x_shape = x.size() if hasattr(x, "size") else [1]
    x_dim = len(x_shape)
    x_dtype = x.dtype if hasattr(x, "dtype") else None
    return x, x_shape, x_dim, x_dtype


def get_op_const_arg_kwarg(const_arg):
    """
    similar to get_op_const_arg_kwarg()
    """
    new_args = const_arg[0]
    len_args = len(new_args)
    assert (
        len_args >= 2 and len_args <= 3
    ), " :currently, op 'Const' support only 2 or 3 params passed!"
    arg0, dtype = new_args[0], new_args[1]
    shape = new_args[2] if len(new_args) == 3 else None
    return arg0, dtype, shape


"""analyze dtype,format"""


def get_cast_dtype(
    type1: Union[str, torch.dtype, type], type2: Union[str, torch.dtype, type]
) -> Union[str, torch.dtype, None]:
    type_map = {
        int: torch.int,
        float: torch.float,
        complex: torch.complex,
        bool: torch.bool,
    }

    type1 = torch.dtype(type1) if isinstance(type1, str) else type1
    type2 = torch.dtype(type2) if isinstance(type2, str) else type2

    type1 = type_map[type1] if isinstance(type1, type) else type1
    type2 = type_map[type2] if isinstance(type2, type) else type2

    if type1 == type2:
        return type1

    complex_list = [torch.complex32, torch.complex64, torch.complex128]
    float_list = [torch.float16, torch.float32, torch.float, torch.float64]
    int_list = [torch.int8, torch.int16, torch.int32, torch.int, torch.int64]

    if type1 in complex_list or type2 in complex_list:
        t1_idx = complex_list.index(type1) if type1 in complex_list else -1
        t2_idx = complex_list.index(type2) if type2 in complex_list else -1
        return complex_list[max(t1_idx, t2_idx)]

    elif type1 == torch.double or type2 == torch.double:
        return torch.double
    elif type1 in float_list or type2 in float_list:
        t1_idx = float_list.index(type1) if type1 in float_list else -1
        t2_idx = float_list.index(type2) if type2 in float_list else -1
        return float_list[max(t1_idx, t2_idx)]
    elif type1 in int_list or type2 in int_list:
        t1_idx = int_list.index(type1) if type1 in int_list else -1
        t2_idx = int_list.index(type2) if type2 in int_list else -1
        return int_list[max(t1_idx, t2_idx)]
    elif type1 == torch.bool or type2 == torch.bool:
        return torch.bool

    assert False, str(type1) + " " + str(type2) + " can't cast these two types!"


def analyze_memory_format(tensor: torch.Tensor, operation: str) -> torch.memory_format:
    original_format = tensor.memory_format

    if operation == "transpose":
        # TODO: transpose
        ...
    elif operation == "permute":
        # TODO: permute
        ...

    return tensor.memory_format if tensor.is_contiguous() else original_format


"""calculate size,stride,storage_offset"""


def get_broadcast_res_two_shape(shape1, shape2) -> Optional[list]:
    len1 = len(shape1)
    len2 = len(shape2)
    max_len = max(len1, len2)
    result_shape = []
    for i in range(-1, -max_len - 1, -1):
        dim1 = shape1[i] if i >= -len1 else 1
        dim2 = shape2[i] if i >= -len2 else 1
        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            result_shape.insert(0, max(dim1, dim2))
        else:
            print(torch.randn(shape1).shape, " ", torch.randn(shape2).shape, end=" ")
            assert False, "input shapes must be broadcastable!"
    return result_shape


def reduce_ops_output_size(
    x_shape, x_dim, dim: Union[None, Sequence, int], keepdim=False
):
    if dim is None or isinstance(dim, Sequence) and len(dim) == 0:
        if keepdim is True:
            shape = [1] * x_dim
        else:
            shape = []  # sum(all) need a scalar as ouput (no shape no stride)
    else:
        dim = [dim] if not isinstance(dim, Sequence) else dim
        dim = [(d + x_dim) % x_dim for d in dim]
        if keepdim is True:
            shape = [1 if r in dim else ori_size for r, ori_size in enumerate(x_shape)]
        else:
            shape = [
                x_shape[r]
                for r in range(x_dim)
                if r not in dim and r - x_dim not in dim
            ]
    return shape


def cal_stride_offset(new_shape: list, offset: list, res: torch.Tensor):
    stride = list(res.stride())
    ori_shape = list(res.size())
    new_offset = 0
    for s, off in zip(stride, offset):
        new_offset += s * off
    stride = [k for k, i, j in zip(stride, ori_shape, new_shape) if i != j]
    return stride, new_offset


"""binary&unary operators"""


def common_binary_op_infer(x1, x2, spec_dtype=None, spec_format=None) -> torch.Tensor:
    x1, x1_shape, x1_dim, x1_dtype = get_fake_tensor_meta_val(x1)
    x2, x2_shape, x2_dim, x2_dtype = get_fake_tensor_meta_val(x2)
    out_shape = get_broadcast_res_two_shape(x1_shape, x2_shape)
    dtype = get_cast_dtype(x1_dtype, x2_dtype) if not spec_dtype else spec_dtype
    memory_format = get_memory_format(x1) if not spec_format else spec_format
    return torch.empty(out_shape, dtype=dtype, memory_format=memory_format)


def common_unary_op_infer(x, spec_dtype=None, spec_format=None) -> torch.Tensor:
    _, x_shape, _, x_dtype = get_fake_tensor_meta_val(x)
    return torch.empty(
        x_shape,
        dtype=x_dtype if not spec_dtype else spec_dtype,
        memory_format=get_memory_format(x) if not spec_format else spec_format,
    )


def reduce_op_infer(x, dims, keepdim) -> torch.tensor:
    x, x_shape, x_dim, x_dtype = get_fake_tensor_meta_val(x)
    out_shape = reduce_ops_output_size(x_shape, x_dim, dims, keepdim)
    return torch.empty(out_shape, dtype=x_dtype, memory_format=get_memory_format(x))


"""other common utils"""


def close2(num, tar=0, rtol=0.00001):
    return math.fabs(num - tar) < rtol
