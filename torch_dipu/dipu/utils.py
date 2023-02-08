
import os
import traceback
import contextlib
import threading
from multiprocessing.util import register_after_fork as _register_after_fork

import torch
import torch._six

import torch_dipu

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False


def is_initialized():
    r"""Returns whether PyTorch's dipu state has been initialized."""
    return _initialized and not _in_bad_fork


def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))


class DeferredDIPUCallError(Exception):
    pass


def init():
    _lazy_init()


# not workered
def _lazy_init():
    pass


def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        torch._C._dipu_set_run_yet_variable_to_false()

_register_after_fork(_after_fork, _after_fork)


def synchronize(device=None):
    r"""Waits for all kernels in all streams on a DIPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch_dipu.dipu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with device(device):
        return torch_dipu._C._dipu_synchronize()


def device_count():
    return torch_dipu._C._dipu_getDeviceCount()


def set_device(device):
    if isinstance(device, str) and 'dipu' in device:
        device = device.replace('dipu', torch_dipu.dipu.native_device)
    if isinstance(device, torch._C.device):
        torch_dipu._C._dipu_setDevice(device.index)
    elif torch.device(device) :
        torch_dipu._C._dipu_setDevice(torch.device(device).index)
    else :
        raise AssertionError("input can not convert to torch.device")


def current_device():
    _lazy_init()
    return torch_dipu._C._dipu_getDevice()

def get_device_name(device_id: int):
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    _lazy_init()
    device_prop = torch_dipu._C._dipu_getDeviceProperties(device_id)
    return device_prop.name

def get_device_properties(device_id: int):
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    _lazy_init()
    return torch_dipu._C._dipu_getDeviceProperties(device_id)

def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch._six.string_classes):
        if "dipu" not in device:
            return int(device)
        else:
            device = torch.device(device)
    device_idx = None
    if isinstance(device, torch._C.device):
        if device.type not in ['dipu', device]:
            raise ValueError('Expected a dipu device, but got: {}'.format(device))
        device_idx = device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch_dipu.dipu.current_device()
        else:
            raise ValueError('Expected a dipu device with a specified index '
                             'or an integer, but got: '.format(device))
    return device_idx


def is_available():
    if (not hasattr(torch_dipu._C, '_dipu_setDevice')):
        return False
    return device_count() > 0


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch_dipu._C._dipu_getDevice()
        if self.prev_idx != self.idx:
            torch_dipu._C._dipu_setDevice(self.idx)
        torch_dipu.dipu._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch_dipu._C._dipu_setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_dipu else -1
        super(device_of, self).__init__(idx)


@contextlib.contextmanager
def stream(stream):
    r"""Context-manager that selects a given stream.

    All DIPU kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    if stream is None:
        yield
        return
    src_prev_stream = current_stream()

    if src_prev_stream.device != stream.device:
        # The given stream is on a different device; have to restore the
        # current_stream on that device on exit as well
        with device(stream.device):
            dst_prev_stream = current_stream()

    torch_dipu._C._dipu_setStream(stream._cdata)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            torch_dipu._C._dipu_setStream(dst_prev_stream._cdata)
        torch_dipu._C._dipu_setStream(src_prev_stream._cdata)


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_dipu.dipu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_dipu.dipu._lazy_init()
    return torch_dipu.dipu.Stream(_cdata=torch_dipu._C._dipu_getCurrentStream(
        _get_device_index(device, optional=True)))


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_dipu.dipu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_dipu.dipu._lazy_init()
    return torch_dipu.dipu.Stream(_cdata=torch_dipu._C._dipu_getDefaultStream(
        _get_device_index(device, optional=True)))


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))
    return type(name, (object,), {"__init__": init_err})


if not hasattr(torch_dipu._C, '_DIPUStreamBase'):
    # Define dummy base classes
    torch_dipu._C.__dict__['_DIPUStreamBase'] = _dummy_type('DIPUStreamBase')
    torch_dipu._C.__dict__['_DIPUEventBase'] = _dummy_type('DIPUEventBase')


def init_dump():
    torch_dipu.dipu._lazy_init()
    return torch_dipu._C._dipu_initDump()

def set_dump(cfg_file):
    torch_dipu.dipu._lazy_init()
    return torch_dipu._C._dipu_setDump(cfg_file)

def finalize_dump():
    torch_dipu.dipu._lazy_init()
    return torch_dipu._C._dipu_finalizeDump()

def get_dipu_overflow_flag():
    float_status = torch.zeros(8).dipu()
    result = torch_dipu.dipu_get_float_status(float_status)
    if (float_status.cpu()[0] != 0):
        return True
    else:
        return False

def clear_dipu_overflow_flag():
    float_status = torch.zeros(8).dipu()
    result = torch_dipu.dipu_clear_float_status(float_status)
