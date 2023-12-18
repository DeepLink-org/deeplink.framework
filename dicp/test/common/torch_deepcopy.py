import functools
import types
from copy import copy
from typing import Any, List, Dict
from importlib import import_module
from contextlib import contextmanager

from torch import nn
from torch._subclasses import FakeTensorMode


def copy_func(func: types.FunctionType, name=None, argdefs=None, closure=None) -> types.FunctionType:
    # from https://stackoverflow.com/a/13503277
    # shallow copy not deep copy
    copied_func = types.FunctionType(func.__code__, func.__globals__,
                                     name=func.__name__ if name is None else name,
                                     argdefs=copy(func.__defaults__) if argdefs is None else argdefs,
                                     closure=copy(func.__closure__) if closure is None else closure)
    copied_func = functools.update_wrapper(copied_func, func)
    copied_func.__kwdefaults__ = copy(func.__kwdefaults__)
    return copied_func


def make_cell(val=None):
    # from https://stackoverflow.com/a/37666086
    x = val

    def closure():
        return x
    return closure.__closure__[0]


def make_closure(closure_dict: Dict[str, Any], keys_list: List[str]):
    return tuple(map(lambda s: make_cell(closure_dict.get(s, None)), keys_list))


# Usage:
# with deepcopy_to_fake_tensor_patched():
#     copied_module = copy.deepcopy(module)
#     fake_tensor_output = copied_module(fake_tensor_input)
@contextmanager
def deepcopy_to_fake_tensor_hf_hook_patched():
    target_module_str = "torch._dynamo.utils"
    origin_func_str = "deepcopy_to_fake_tensor"

    target_module = import_module(target_module_str)
    origin_func = getattr(target_module, origin_func_str)

    def patch_module(copied_obj: nn.Module, obj: nn.Module):
        if hasattr(obj, "_hf_hook") and hasattr(obj, "_old_forward") \
                and not hasattr(copied_obj, "_hf_hook_forward_patched"):
            new_closure_dict = {
                "module": copied_obj,
                "old_forward": functools.partial(copied_obj.__class__.forward, copied_obj),
            }
            new_closure_keys_list = copied_obj.forward.__code__.co_freevars
            new_closure = make_closure(new_closure_dict, new_closure_keys_list)
            new_forward_func = copy_func(copied_obj.forward, closure=new_closure)
            copied_obj._hf_hook_forward_patched = True
            copied_obj.forward = new_forward_func

        for name, child in obj.named_children():
            patch_module(getattr(copied_obj, name), child)

    @functools.wraps(origin_func)
    def deepcopy_to_fake_tensor_patched(obj: Any, fake_mode: FakeTensorMode):
        copied_obj = origin_func(obj, fake_mode)
        patch_module(copied_obj, obj)
        return copied_obj

    setattr(target_module, origin_func_str, deepcopy_to_fake_tensor_patched)

    yield

    setattr(target_module, origin_func_str, origin_func)
