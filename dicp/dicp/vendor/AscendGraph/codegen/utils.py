import enum
import acl
import torch


@enum.unique
class AclDataType(enum.Enum):
    ACL_DT_UNDEFINED = -1
    ACL_FLOAT = 0
    ACL_FLOAT16 = 1
    ACL_INT8 = 2
    ACL_INT32 = 3
    ACL_UINT8 = 4
    ACL_INT16 = 6
    ACL_UINT16 = 7
    ACL_UINT32 = 8
    ACL_INT64 = 9
    ACL_UINT64 = 10
    ACL_DOUBLE = 11
    ACL_BOOL = 12
    ACL_STRING = 13
    ACL_COMPLEX64 = 16
    ACL_COMPLEX128 = 17
    ACL_BF16 = 27
    ACL_UINT1 = 30
    ACL_COMPLEX32 = 33


@enum.unique
class AclFormat(enum.Enum):
    ACL_FORMAT_UNDEFINED = -1
    ACL_FORMAT_NCHW = 0
    ACL_FORMAT_NHWC = 1
    ACL_FORMAT_ND = 2
    ACL_FORMAT_NC1HWC0 = 3
    ACL_FORMAT_FRACTAL_Z = 4
    ACL_FORMAT_NC1HWC0_C04 = 12
    ACL_FORMAT_HWCN = 16
    ACL_FORMAT_NDHWC = 27
    ACL_FORMAT_FRACTAL_NZ = 29
    ACL_FORMAT_NCDHW = 30
    ACL_FORMAT_NDC1HWC0 = 32
    ACL_FRACTAL_Z_3D = 33


def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


def get_acl_format(x) -> int:
    if hasattr(x, 'meta') and 'native_memory_format' in x.meta:
        return AclFormat[x.meta['native_memory_format']].value
    else:
        return AclFormat.ACL_FORMAT_ND.value


def get_acl_dtype(dtype: torch.dtype) ->int:
    if dtype == torch.bool:
        return AclDataType.ACL_BOOL.value
    elif dtype == torch.int64:
        return AclDataType.ACL_INT64.value
    elif dtype in [torch.float32, torch.float]:
        return AclDataType.ACL_FLOAT.value
    elif dtype == torch.float16:
        return AclDataType.ACL_FLOAT16.value
    elif dtype == torch.int32:
        return AclDataType.ACL_INT32.value
    elif dtype == torch.complex64:
        return AclDataType.ACL_COMPLEX64.value
    elif dtype == torch.bfloat16:
        return AclDataType.ACL_BF16.value
    else:
        raise RuntimeError(f"unknow torch data type ({dtype}) in get_acl_dtype!")


def get_torch_dtype(d: int) -> torch.dtype:
    if d == AclDataType.ACL_BOOL.value:
        return torch.bool
    elif d == AclDataType.ACL_INT64.value:
        return torch.int64
    elif d == AclDataType.ACL_FLOAT.value:
        return torch.float32
    elif d == AclDataType.ACL_FLOAT16.value:
        return torch.float16
    elif d == AclDataType.ACL_INT32.value:
        return torch.int32
    elif d == AclDataType.ACL_COMPLEX64.value:
        return torch.complex64
    elif d == AclDataType.ACL_BF16.value:
        return torch.bfloat16
    else:
        raise RuntimeError(f"unknow acl data type ({d}) in get_torch_dtype!")


def get_shape_from_desc(desc) -> list:
    shape = []
    dims = acl.get_tensor_desc_num_dims(desc)
    for i in range(dims):
        shape.append(acl.get_tensor_desc_dim_v2(desc, i)[0])
    return shape


def symint_in_shape(shape):
    for elem in shape:
        if isinstance(elem, torch.SymInt):
            return True
    return False


def get_ascend_dtype_num(dtype: str):
    if dtype == "FLOAT":
        return AclDataType.ACL_FLOAT.value
    elif dtype == "FLOAT16":
        return AclDataType.ACL_FLOAT16.value
    elif dtype == "INT32":
        return AclDataType.ACL_INT32.value
    elif dtype == "INT64":
        return AclDataType.ACL_INT64.value
    elif dtype == "BOOL":
        return AclDataType.ACL_BOOL.value
    elif dtype == "COMPLEX64":
        return AclDataType.ACL_COMPLEX64.value
    elif dtype == "UINT1":
        return AclDataType.ACL_UINT1.value
    elif dtype == "UINT8":
        return AclDataType.ACL_UINT8.value
    elif dtype == "UINT64":
        return AclDataType.ACL_UINT64.value
    elif dtype == "BF16":
        return AclDataType.ACL_BF16.value
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

