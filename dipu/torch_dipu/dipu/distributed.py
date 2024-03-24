from datetime import timedelta

import torch
from torch import distributed as dist
from torch.distributed import (
    Backend,
    Store,
    default_pg_timeout,
    ProcessGroup,
    ProcessGroupGloo,
)
from typing import Any, Optional, Union

from torch_dipu import mockcuda
from torch_dipu import dipu
from torch_dipu import _C

dicl_backend = _C.dicl_backend
ProcessGroupDICL = _C.ProcessGroupDICL


def reg_dicl(store, rank, size, timeout):
    return ProcessGroupDICL(store, rank, size, timeout)


if dipu.get_dipu_torch_version() == dipu.torch_ver_200:
    Backend.register_backend(dicl_backend, reg_dicl)
else:
    Backend.register_backend(
        dicl_backend, reg_dicl, devices="cuda" if mockcuda else dipu.diputype
    )


# distributed.BackendConfig has no power to do suitable 'device_backend_map' setting
# so we use this patch to let cpu use gloo backend.
_raw_register_backend = ProcessGroup._register_backend


def _wrapped_register_backend(self, device, backend_type, backend):
    # dicl not support cpu tensor
    if device.type == "cpu" and isinstance(backend, ProcessGroupDICL):
        print(" try create a gloo backend,",  backend.size(), flush=True)
        backend = ProcessGroupGloo(
            backend.store(), backend.rank(), backend.size(), timeout=backend.timeout()
        )
        backend_type = ProcessGroup.BackendType.GLOO

    # if mock_cuda=true and "DIPU_PYTHON_DEVICE_AS_CUDA" = 'True'. the python layer
    # device.cuda is actually device.xpu(dipu) in cpp layer, so this func reg a
    # backend to xpu(dipu), but if the backend is gloo or mpi which not support dipu.
    # call of any comm func (except pg.barrier()) on xpu tensor will fail, it same as
    # the original meaning of gloo pg which not support device tensor.
    _raw_register_backend(self, device, backend_type, backend)


# change nccl to internal used dicl, so existing model can keep nccl as backend name.
_raw_init_process_group = dist.init_process_group


def _wrap_init_process_groups(
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
    _raw_init_process_group(
        backend, init_method, timeout, world_size, rank, store, group_name, pg_options
    )


# seems only mmcv use this func to determine backend, so we return nccl
# if backend it's dicl. or do we need to change _world.pg_map value?
_raw_get_backend = dist.get_backend


def _wrap_get_backend(group: Optional[ProcessGroup] = None) -> str:
    ret = _raw_get_backend(group)
    if ret == dicl_backend and mockcuda:
        return Backend.NCCL
    else:
        return ret

# dicl not support coalescing now. Todo: remove after support
def wrap_batch_isend_irecv(p2p_op_list):
    dist.distributed_c10d._check_p2p_op_list(p2p_op_list)
    reqs = []
    for p2p_op in p2p_op_list:
        work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        if work:
            reqs.append(work)
    return reqs

# huawei AscendSpeed pass rank list like [0, 0], which cause pg 
# creation fail in torch 2.0. actually it's huawei's problem, such input
# is not valid, but nothing else we can do.
# torch 2.1 handle duplication rank inner
_raw_new_group = dist.new_group
def wrap_new_group(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None):
    ranks = list(set(ranks))  # dedup
    return _raw_new_group(ranks, timeout, backend, pg_options)


def apply_dist_patch():
    dist.get_backend = _wrap_get_backend
    dist.init_process_group = _wrap_init_process_groups
    dist.ProcessGroup._register_backend = _wrapped_register_backend
    dist.batch_isend_irecv = wrap_batch_isend_irecv

    if dipu.get_dipu_torch_version() == dipu.torch_ver_200:
        dist.new_group = wrap_new_group
