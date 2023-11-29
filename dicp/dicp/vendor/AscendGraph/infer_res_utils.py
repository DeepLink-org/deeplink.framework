from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch

"""parse and get val"""


def get_fake_tensor_meta_val(
    x, req_dim=True, req_dtype=True
) -> Tuple[torch.Tensor, Union[torch.Size, list], int, Union[torch.dtype, type, None]]:
    x_shape = x.size() if hasattr(x, "size") else [1]
    x_dim = len(x_shape)
    x_dtype = x.dtype if hasattr(x, "dtype") else None
    return x, x_shape, x_dim, x_dtype


def get_op_const_arg_kwarg(const_arg):
    """
    if some operator uses Const as an input, call this func to get the input (args and kwargs) of the input op.
    Some operators like "reshape" need a tensor's value(shape), so for operators like "Const" we directly pass its input
    (including value and shape) instead of constructing a fakeTensor, which will neglect a tensor's value.
    input:
        - const_arg: Tuple (new_args,kwargs)
            - new_args: Tuple, identical to input-"new_args" of operator Const
            - kwargs: dict, identical to input-"kwargs" of operator Const

    output:
        - arg0: list, value of "Const"'s input
        - arg2: list, shape of "Const"'s input
    """
    new_args = const_arg[0]
    arg0 = new_args[0]
    arg2 = new_args[2]
    return arg0, arg2


def get_op_const_arg_kwarg(const_arg):
    """
    similar to get_op_const_arg_kwarg()
    """
    new_args = const_arg[0]
    shape = new_args[0]
    dim = new_args[2]
    return shape, dim


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
            shape = []
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
