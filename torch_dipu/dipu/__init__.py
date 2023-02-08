__all__ = [
    "device", "vendor_device", "is_initialized", "_lazy_call", "_lazy_init", "init",
    "synchronize", "device_count", "set_device", "current_device", "get_device_name",
    "get_device_properties", "_get_device_index", "is_available", "device", "device_of",
    "stream", "current_stream", "default_stream", "manual_seed", "manual_seed_all",
    "seed", "seed_all", "initial_seed", "_free_mutex", "caching_allocator_alloc",
    "caching_allocator_delete", "empty_cache", "Stream", "Event"
]

import torch

from .device import __diputype__ as diputype
from .device import __vendor__ as vendor_type
from .device import _device
from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
                    synchronize, device_count, set_device, current_device, get_device_name,
                    get_device_properties, _get_device_index, is_available, device, device_of,
                    stream, current_stream, default_stream)
from .random import manual_seed, manual_seed_all, seed, seed_all, initial_seed
from .memory import (_free_mutex, caching_allocator_alloc, caching_allocator_delete,
                     empty_cache)
from .streams import Stream, Event


def apply_rt_patch():
    torch.device = _device

def apply_class_patches():
    apply_rt_patch()

apply_class_patches()