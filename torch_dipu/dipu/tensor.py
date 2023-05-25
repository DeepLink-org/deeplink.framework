# Copyright (c) 2023, DeepLink.
import torch
from .device import __diputype__


# replace type exported from csrc/tensor/python_tensor.cpp
# need replace torch._tensor_classes? (seems currently not need)
class _MetaTensorType(type):
    def __instancecheck__(cls, inst):
        if isinstance(inst, torch.Tensor):
          if inst.device.type == __diputype__ and inst.dtype == cls.dtype:
            return True
        return False

class FloatTensor(metaclass=_MetaTensorType):
   dtype = torch.float
class DoubleTensor(metaclass=_MetaTensorType):
   dtype= torch.float64
class HalfTensor(metaclass=_MetaTensorType):
   dtype= torch.float16

class LongTensor(metaclass=_MetaTensorType):
    dtype = torch.int64
class IntTensor(metaclass=_MetaTensorType):
    dtype = torch.int32
class ShortTensor(metaclass=_MetaTensorType):
    dtype = torch.int16
class ByteTensor(metaclass=_MetaTensorType):
    dtype = torch.uint8
class CharTensor(metaclass=_MetaTensorType):
    dtype = torch.int8
class BoolTensor(metaclass=_MetaTensorType):
    dtype = torch.bool
