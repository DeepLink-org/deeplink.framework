from datetime import timedelta

import torch
from torch import distributed as dist
from torch.distributed import Backend, Store, default_pg_timeout, ProcessGroup, ProcessGroupGloo
from typing import Any, Optional, Union

from torch_dipu import mockcuda
from torch_dipu import _C
dicl_backend = _C.dicl_backend
ProcessGroupDICL = _C.ProcessGroupDICL

def reg_dicl(store, rank, size, timeout):
    return ProcessGroupDICL(store, rank, size, timeout)

Backend.register_backend(dicl_backend, reg_dicl)


# distributed.BackendConfig has no power to do suitable 'device_backend_map' setting 
# so we use this patch to let cpu use gloo backend.
_raw_register_backend = ProcessGroup._register_backend
def _wrapped_register_backend(self, device, backend_type, backend):
    # dicl not support cpu tensor
    if device.type == "cpu" and isinstance(backend, ProcessGroupDICL):
      backend = ProcessGroupGloo(backend.store(), backend.rank(), backend.size(), timeout=backend.timeout())
      backend_type = ProcessGroup.BackendType.GLOO

    _raw_register_backend(self, device, backend_type, backend)


_raw_init_process_group = dist.init_process_group
def _init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: timedelta = default_pg_timeout,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
):
  if backend == None or (backend == Backend.NCCL and mockcuda):
    backend = dicl_backend
  _raw_init_process_group(backend, init_method, timeout, 
                          world_size, rank, store, group_name, pg_options)


# seems only mmcv use this func to determine backend
# do we need do change _world.pg_map value when set ?
_raw_get_backend = dist.get_backend
def _get_backend(group: Optional[ProcessGroup] = None) -> str:
   ret = _raw_get_backend(group)
   if ret == dicl_backend and mockcuda:
       return Backend.NCCL
   else:
      return ret

def apply_dist_patch():
  dist.get_backend = _get_backend
  dist.init_process_group = _init_process_group
  dist.ProcessGroup._register_backend = _wrapped_register_backend