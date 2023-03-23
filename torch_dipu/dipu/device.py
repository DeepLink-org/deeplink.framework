import torch
from torch import device
from torch_dipu import mockcuda
__dipu__ = 'dipu'
__diputype__ = 'privateuseone'
__vendor__ = 'mlu'  # need update when compile


class MetaType(type):
    device = torch.device
    def __instancecheck__(self, instance):
        if isinstance(instance, MetaType.device):
            return True
        return False

# torch.Device is a final class. cannot inherit
# csrc/Device.cpp THPDevice_pynew:
# "Device(Device device)" Device type can be Device, Long, String
# "Device(c10::string_view type, int64_t? index=-1)"
class _DeviceWrapper(metaclass=MetaType):
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
            return MetaType.device(dev_name)
        # handle device as str
        if len(args) >= 1 and isinstance(args[0], str):
            argList = list(args)
            argList[0] = cls.__doreplace(args[0])
            args = tuple(argList)
        # handle device in type key, not support int type but str and device
        deviceValue = kwargs.get("type", None)
        if deviceValue != None and isinstance(deviceValue, str):
            kwargs["type"] = cls.__doreplace(deviceValue)
        return MetaType.device(*args, **kwargs)


# wrap device related func
def GetDeviceProxy(rawfunc, static_func = False, pos = 0, name = "device"):
    def _replaceDevice(args, kwargs):
        # pos device
        if len(args) >= pos+1 and (isinstance(args[pos], int)
                or isinstance(args[pos], str)):
            argList = list(args)
            argList[pos] = torch.device(args[pos])
            args = tuple(argList)
        deviceValue = kwargs.get(name, None)
        if deviceValue != None:
            kwargs[deviceValue] = torch.device(deviceValue)
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
