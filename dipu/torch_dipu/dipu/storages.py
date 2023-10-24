# Copyright (c) 2023, DeepLink.
import torch
from torch.serialization import register_package
from torch.storage import UntypedStorage
from torch_dipu import mockcuda
from .device import __diputype__, __dipu__, _get_device_index
from .device import *
from torch_dipu import _C

def __validate_dipu_device(location):
  device_idx = _get_device_index(location, True)

  if not is_available():
      raise RuntimeError('Attempting to deserialize object on a DIPU '
                          'device but dipu is_available() is False. '
                          'If you are running on a CPU-only machine, '
                          'please use torch.load with map_location=torch.device(\'cpu\') '
                          'to map your storages to the CPU.')
  cnt = device_count()
  if device_idx >= cnt:
      raise RuntimeError('Attempting to deserialize object on DIPU device '
                          f'{device_idx} but dipu.device_count() is {cnt}. Please use '
                          'torch.load with map_location to map your storages '
                          'to an existing device.')
  return device_idx

# this obj is storage
def _dipu_deserialize(obj, location):
  if location.startswith(__diputype__) or location.startswith(__dipu__) or \
          (mockcuda and location.startswith("cuda")):
      device_idx = __validate_dipu_device(location)
      if getattr(obj, "_torch_load_uninitialized", False):
          with devicectx(device_idx):
              return torch.UntypedStorage(obj.nbytes(), device= torch.device(location))
      else:
          return obj.dipu(device_idx)

# this obj is storage, so it's device.type is real diputype (may be cuda or xpu depend on if set DIPU_PYTHON_DEVICE_AS_CUDA)
def _dipu_tag(obj):
  if obj.device.type == __diputype__:
      return __diputype__ + str(obj.device.index)

# cuda is 20
register_package(15, _dipu_tag, _dipu_deserialize)

# handle Storage dipu 
def _dipu_storage(self, device=None, non_blocking=False, **kwargs):
  if self.is_dipu:
      if device is None:
          device = current_device()
      if self.get_device() == device:
          return self
  else:
      if device is None:
          device = -1

  with devicectx(device):
    untyped_storage = torch.UntypedStorage(
        self.size(), device=torch.device(__diputype__)
    )
    untyped_storage.copy_(self, non_blocking)
    return untyped_storage

@property
def _is_dipu_storage(self):
    return self.device.type == __diputype__

setattr(UntypedStorage, 'is_dipu', _is_dipu_storage)
UntypedStorage.dipu = _dipu_storage

if mockcuda:
  UntypedStorage.is_cuda = UntypedStorage.is_dipu
  UntypedStorage.cuda = UntypedStorage.dipu 

_raw_storage_resize = UntypedStorage.resize_
def _resize(self, size: int):
   if self.device.type != __diputype__:
      return _raw_storage_resize(self, size)
   else:
      return _C.storage_resize_(self, size)
  
UntypedStorage.resize_ = _resize

UntypedStorage.__new__ = GetDeviceProxy(torch.UntypedStorage.__new__, pos = -1, caller="class_new") 

