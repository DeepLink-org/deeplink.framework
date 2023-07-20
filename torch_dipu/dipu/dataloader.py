import torch
from torch.utils.data import (DataLoader, Sampler, Dataset)

from typing import Any, Callable, Iterable, TypeVar, Sequence, List, Optional, Union

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

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