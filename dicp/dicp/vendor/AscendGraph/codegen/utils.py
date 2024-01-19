import torch


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def get_ascend_dtype_num(dtype: str):
    if dtype == "FLOAT":
        return 0
    elif dtype == "FLOAT16":
        return 1
    elif dtype == "INT32":
        return 3
    elif dtype == "INT64":
        return 9
    elif dtype == "BOOL":
        return 12
    elif dtype == "COMPLEX64":
        return 16
    elif dtype == "UINT1":
        return 30
    elif dtype == "UINT8":
        return 4
    elif dtype == "UINT64":
        return 10
    elif dtype == "BF16":
        return 27
    else:
        raise RuntimeError(f"unknow torch data type ({dtype}) in get_ascend_dtype_num!")


def get_ascend_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bool:
        return "BOOL"
    elif dtype == torch.int64:
        return "INT64"
    elif dtype in [torch.float32, torch.float]:
        return "FLOAT"
    elif dtype == torch.float16:
        return "FLOAT16"
    elif dtype == torch.int32:
        return "INT32"
    elif dtype == torch.complex64:
        return "COMPLEX64"
    elif dtype == torch.bfloat16:
        return "BF16"
    else:
        raise RuntimeError(f"unknow torch data type ({dtype}) in get_ascend_dtype!")


def get_cpp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.int64:
        return "INT64"
    elif dtype == torch.float32:
        return "FLOAT"
    elif dtype == torch.int32:
        return "INT32"
    else:
        raise RuntimeError(f"unknow torch data type ({dtype}) in get_cpp_dtype!")
