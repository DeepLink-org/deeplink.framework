# Copyright (c) 2023, DeepLink.
from .utils import is_initialized
from .device import __diputype__ as diputype
from .device import __vendor__ as vendor_type
from .device import *
from .random_dipu import *
from .memory import *
from .streams import *
from .tensor import *
from .storages import *
import torch_dipu

_is_in_bad_fork = getattr(torch_dipu._C, "_is_in_bad_fork", lambda: False)

# DIPU need follow api in https://pytorch.org/docs/stable/cuda.html, but shoudln't appear name
# as "gpu" or "cuda" (mock cuda is another problem)
# only partially aligned now,
__all__ = [
    # resume initialize flag after random generator ready
    # "is_initialized",

    # tensor
    'FloatTensor', 'DoubleTensor', 'HalfTensor',
    'LongTensor', 'IntTensor', 'ShortTensor', 'ByteTensor', 'CharTensor', 'BoolTensor',

    # device
    "can_device_access_peer",  "current_device",  "device", "device_count", "device_of", "synchronize",
    "get_device_name", "get_device_properties", "get_device_capability", "is_available", "set_device",
    "GetDeviceProxy", "GetDeviceStaticProxy", "diputype", "vendor_type",

    # stream
    "current_stream", "default_stream", "set_stream", "set_sync_debug_mode", "stream", "StreamContext", "Stream", "Event",

    # random
    "get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all",
    "manual_seed", "manual_seed_all", "seed", "seed_all", "initial_seed",
    "_is_in_bad_fork",
    # # mem manage
    "reset_peak_memory_stats", "empty_cache", "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved",
    # "caching_allocator_alloc", "caching_allocator_delete", "memory_summary", "memory_stats"

    # not support mock cuda_graph now
]

import atexit
atexit.register(release_all_resources)