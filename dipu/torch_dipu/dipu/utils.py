# Copyright (c) 2023, DeepLink.

import os
import traceback
import threading
from multiprocessing.util import register_after_fork as _register_after_fork
import torch
from torch_dipu import _C

_initialized = True
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False


def format_cast(tensor:torch.Tensor, format:_C.DIOPIMemoryFormat) -> torch.Tensor:
    return _C.format_cast(tensor, format)

def is_initialized():
    r"""Returns whether PyTorch's dipu state has been initialized."""
    return _initialized and not _in_bad_fork

def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))

def _lazy_init():
    pass

def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        # torch._C._dipu_set_run_yet_variable_to_false()

_register_after_fork(_after_fork, _after_fork)

def _dummy_type(name: str) -> type:
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(
                "Tried to instantiate dummy base class {}".format(class_name))
        return err_fn
    return type(name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)})


