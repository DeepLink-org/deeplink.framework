# Copyright (c) 2023, DeepLink.
from functools import partial
from typing import Optional, Tuple, Union

import torch

from torch_dipu import mockcuda
from torch_dipu import _C
__dipu__ = 'dipu'
__diputype__ = 'xpu'
__vendor__ = _C.dipu_vendor  # need update when compile
_device_t = Union[torch.device, str, int, None]

class _MetaDeviceType(type):
    device_ = torch.device
    def __instancecheck__(cls, instance):
        if isinstance(instance, _MetaDeviceType.device_):
            return True
        return False

# torch.Device is a final class. cannot inherit
# csrc/Device.cpp THPDevice_pynew:
# "Device(Device device)" Device type can be Device, Long, String
# "Device(c10::string_view type, int64_t? index=-1)"
class _DIPUDevice(metaclass=_MetaDeviceType):
    @staticmethod
    def __doreplace(arg):
        if (__dipu__ in arg):
            arg = arg.replace(__dipu__, __diputype__)
        if (mockcuda and "cuda" in arg):
            arg = arg.replace("cuda", __diputype__)
        return arg

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], int) and mockcuda:
            # modify default int device type only when "mock cuda".
            dev_name = __diputype__ + ":" + str(args[0])
            return _MetaDeviceType.device_(dev_name)
        # handle device as str
        if len(args) >= 1 and isinstance(args[0], str):
            argList = list(args)
            argList[0] = cls.__doreplace(args[0])
            args = tuple(argList)
        # handle device in type key, not support int type but str and device
        deviceValue = kwargs.get("type", None)
        if deviceValue != None and isinstance(deviceValue, str):
            kwargs["type"] = cls.__doreplace(deviceValue)
        return _MetaDeviceType.device_(*args, **kwargs)

torch.device = _DIPUDevice

# wrap device related func
def GetDeviceProxy(rawfunc, pos = 0, name = "device", static_func = False):
    def _replaceDevice(args, kwargs):
        # pos device
        if pos >= 0 and pos < len(args) and (isinstance(args[pos], int)
                or isinstance(args[pos], str)):
            argList = list(args)
            argList[pos] = torch.device(args[pos])
            args = tuple(argList)
        deviceValue = kwargs.get(name, None)
        if deviceValue != None and (isinstance(deviceValue, int)
                or isinstance(deviceValue, str)):
            kwargs[name] = torch.device(deviceValue)
        return args, kwargs

    def _proxyFuncInst(self, *args, **kwargs):
        args, kwargs = _replaceDevice(args, kwargs)
        return rawfunc(self, *args, **kwargs)
    
    def _proxyFuncStatic(*args, **kwargs):
        args, kwargs = _replaceDevice(args, kwargs)
        return rawfunc(*args, **kwargs)
    if static_func:
        return _proxyFuncStatic
    else:
        return _proxyFuncInst

GetDeviceStaticProxy = partial(GetDeviceProxy, pos = -1, name = "device", static_func = True)

def _lazy_init():
    pass

# dipu device Interface

def device_count():
    return _C._dipu_get_device_count()

def set_device(device):
    _lazy_init()
    if isinstance(device, torch.device):
        _C._dipu_set_device(device.index)
    elif (isinstance(device, int) or isinstance(device, str)):
        _C._dipu_set_device(torch.device(device).index)
    else :
        raise AssertionError("input can not convert to torch.device")

def current_device():
    _lazy_init()
    return _C._dipu_current_device()

def _get_device_index(device, optional=False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    # _DIPUDevice not support bytes replace now, is it really needed?
    if isinstance(device, str):
        device = torch.device(device)
    device_idx = None
    if isinstance(device, torch.device):
        if device.type not in [__diputype__]:
            raise ValueError('Expected a dipu device, but got: {}'.format(device))
        device_idx = device.index
    elif isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return current_device()
        else:
            raise ValueError('Expected a dipu device with a specified index '
                             'or an integer, but got: '.format(device))
    return device_idx

def synchronize(_device=None):
    r"""Waits for all kernels in all streams on a DIPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with device(_device):
        return _C._dipu_synchronize()


def is_available():
    if (not hasattr(_C, '_dipu_set_device')):
        return False
    return device_count() > 0


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = current_device()
        if self.prev_idx != self.idx:
            set_device(self.idx)

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            set_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """
    def __init__(self, obj):
        idx = obj.get_device() if obj.is_dipu else -1
        super(device_of, self).__init__(idx)


# device properties.
def can_device_access_peer():
    return False

def get_device_name(device: Optional[_device_t] = None) -> str:
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: _device_t) -> _C._DIPUDeviceProperties:
    _lazy_init()
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    return _C._dipu_getDeviceProperties(device_id)
