from torch_dipu import _C


def get_autocast_dipu_dtype():
    return _C.get_autocast_dipu_dtype()


def is_autocast_dipu_enabled():
    return _C.is_autocast_dipu_enabled()


def set_autocast_dipu_enabled(enabled):
    return _C.set_autocast_dipu_enabled(enabled)


def set_autocast_dipu_dtype(dtype):
    return _C.set_autocast_dipu_dtype(dtype)


def is_bf16_supported():
    return True
