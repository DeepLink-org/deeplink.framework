import torch
from torch.utils.data import (DataLoader, Sampler, Dataset)

from typing import Any, Callable, Iterable, TypeVar, Sequence, List, Optional, Union

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

# When we create a DataLoader and set pin_memory=True,
# this will create a thread to perform the pin_memory operation on the tensor.
# The newly created thread will have device 0 as its default device,
# which will result in initializing the context of device 0 and occupying its memory.
# This will cause a shortage of memory on device 0 in a multi-GPU scenario.
# To address this issue, we introduce the DIPUDataLoader and monkey patch torch.utils.data.DataLoader.
# When the pin_memory parameter is set to True, we set pin_memory_device to 'cuda'.
# The pin_memory thread generated will set current device to main thread's device index
# instead of using the default 0th device thus avoid occupying device 0 memory.

# As for why native Pytorch does not have this issue on Cuda,
# The CUDAHostAllocator performs an operation before allocating memory and obtains a primary ctx.
# Depending on whether the primary ctx flag is set, the device set by the main thread can be obtained.
# So even if the current of the child thread's device is 0,
# the CUDAHostAllocator can also obtain the device of the main thread and set as allocator's own device.
# DIPU currently does not have such logic and supports this feature requires API similar to cuda's cuDevicePrimaryCtxGetState.
class DIPUDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                pin_memory: bool = False, drop_last: bool = False,
                timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                multiprocessing_context=None, generator=None,
                *, prefetch_factor: Optional[int] = None,
                persistent_workers: bool = False,
                pin_memory_device: str = ""):
        if pin_memory:
            pin_memory_device = 'cuda'

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
                         collate_fn, pin_memory, drop_last, timeout, worker_init_fn,
                         multiprocessing_context, generator,
                         prefetch_factor = prefetch_factor,
                         persistent_workers = persistent_workers,
                         pin_memory_device = pin_memory_device)

def apply_dataloader_patch():
    torch.utils.data.DataLoader = DIPUDataLoader