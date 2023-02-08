import torch_dipu

from .utils import _lazy_init, _lazy_call, device_count, current_device

__all__ = ['manual_seed', 'manual_seed_all',
           'seed', 'seed_all', 'initial_seed']


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
        default_generator = torch_dipu.dipu.default_generators[idx]
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
            default_generator = torch_dipu.dipu.default_generators[i]
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
        default_generator = torch_dipu.dipu.default_generators[idx]
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
            default_generator = torch_dipu.dipu.default_generators[i]
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
    default_generator = torch_dipu.dipu.default_generators[idx]
    return default_generator.initial_seed()
