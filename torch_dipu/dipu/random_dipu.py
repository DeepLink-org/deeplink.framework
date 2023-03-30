from typing import Iterable, List, Union
import torch
from torch import Tensor

from torch_dipu import _C
from .utils import _lazy_init, _lazy_call
from .device import device_count, current_device, _get_device_index


### Random sampling
def get_rng_state(device: Union[int, str, torch.device] = 'dipu') -> Tensor:
    r"""Returns the random number generator state of the specified DIPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'dipu'`` (i.e., ``torch.device('dipu')``, the current dipu device).

    """
    idx = _get_device_index(device)
    if idx is None:
        idx = current_device()
    return _C._get_rng_state(idx)


def get_rng_state_all() -> List[Tensor]:
    r"""Returns a list of ByteTensor representing the random number states of all devices."""
    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results


def set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = 'dipu') -> None:
    r"""Sets the random number generator state of the specified DIPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'dipu'`` (i.e., ``torch.device('dipu')``, the current DIPU device).

    .. warning::
        The state of GPU or CPU is different with DIPU, setting GPU state to DIPU does not
        come into effect.
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('dipu', device)

    idx = device.index
    if idx is None:
        idx = current_device()

    _C._set_rng_state(idx, new_state_copy)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Sets the random number generator state of all devices.

    Args:
        new_state (Iterable of torch.ByteTensor): The desired state for each device"""
    for i, state in enumerate(new_states):
        set_rng_state(state, i)



def manual_seed(seed):
    r"""Sets the seed for generating random numbers for the current dipu.
    It's safe to call this function if dipu is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-dipu model, this function is insufficient
        to get determinism.  To seed all dipus, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = _C.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all dipus.
    It's safe to call this function if dipu is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = _C.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb)


def seed():
    r"""Sets the seed for generating random numbers to a random number for the current dipu.
    It's safe to call this function if dipu is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-dipu model, this function will only initialize
        the seed on one dipu.  To initialize all dipus, use :func:`seed_all`.
    """
    def cb():
        idx = current_device()
        default_generator = _C.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all dipus.
    It's safe to call this function if dipu is not available; in that
    case, it is silently ignored.
    """
    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = _C.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed():
    r"""Returns the current random seed of the current dipu.

    .. warning::
        This function eagerly initializes dipu.
    """
    _lazy_init()
    idx = current_device()
    default_generator = _C.default_generators[idx]
    return default_generator.initial_seed()
