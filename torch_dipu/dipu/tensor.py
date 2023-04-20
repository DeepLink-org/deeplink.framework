import torch
from .device import __diputype__

class DIPUTensorType:
    def __init__(self, type: torch.dtype):
       self.type = type
       
    def __instancecheck__(self, inst):
        if isinstance(inst, torch.Tensor):
          if inst.device.type == __diputype__ and inst.dtype == self.type:
            return True
        return False


FloatTensor = DIPUTensorType(torch.float)
DoubleTensor = DIPUTensorType(torch.float64)
HalfTensor = DIPUTensorType(torch.float16)

LongTensor = DIPUTensorType(torch.int64)
IntTensor = DIPUTensorType(torch.int32)
ShortTensor = DIPUTensorType(torch.int16)
ByteTensor = DIPUTensorType(torch.uint8)
CharTensor = DIPUTensorType(torch.int8)
BoolTensor = DIPUTensorType(torch.bool)
