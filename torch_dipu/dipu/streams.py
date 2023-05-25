# Copyright (c) 2023, DeepLink.

import ctypes
import contextlib
from typing import Union, Optional, Any
import torch
from torch.types import _int

from torch_dipu import _C
from .utils import _dummy_type
from .device import _get_device_index, _lazy_init
from .device import device as dipu_device
if not hasattr(_C, '_DIPUStreamBase'):
    # Define dummy base classes
    torch._C.__dict__['_DIPUStreamBase'] = _dummy_type('_DIPUStreamBase')
    torch._C.__dict__['_DIPUEventBase'] = _dummy_type('_DIPUEventBase')


class Stream(_C._DIPUStreamBase):
    r"""Wrapper around a dipu stream.

    A dipu stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`dipu-semantics` for
    details.

    Arguments:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream. Lower numbers
                                 represent higher priorities.
    """

    def __init__(self, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            super(Stream, self).__init__(priority=priority, **kwargs)
        else:
            with dipu_device(device):
                super(Stream, self).__init__(priority=priority, **kwargs)
        return

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Arguments:
            event (Event): an event to wait for.

        .. note:: This is a wrapper around ``dipuStreamWaitEvent()``

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Arguments:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Arguments:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super(Stream, self).query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``dipuStreamSynchronize()``: see
           `dipu Stream documentation`_ for more info.
        """
        super(Stream, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.dipu_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.dipu_stream, self.device))

    def __repr__(self):
        return ('<torch_dipu.dipu.Stream device={0} dipu_stream={1:#x}>'
                .format(self.device, self.dipu_stream))
    # mock
    @property
    def priority(self):
        return 0  


def set_stream(stream: Stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.
    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _lazy_init()
    _C._dipu_setStream(stream_id=stream.stream_id, device_index=stream.device_index)

def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_dipu.dipu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    st = _C._dipu_getCurrentStream(_get_device_index(device, optional=True))
    return Stream(stream_id=st.stream_id, device_index=st.device_index, device_type=st.device_type)


def default_stream(device=None):
    _lazy_init()
    st = _C._dipu_getDefaultStream(_get_device_index(device, optional=True))
    return Stream(stream_id=st.stream_id, device_index=st.device_index, device_type=st.device_type) 

def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    pass

class StreamContext:
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
    cur_stream : Optional['torch_dipu.dipu.Stream']

    def __init__(self, stream):
        self.stream = stream
        self.idx = _get_device_index(None, True)
    
    def __enter__(self):
          # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or DIPU device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with dipu_device(cur_stream.device):
                self.dst_prev_stream = current_stream(cur_stream.device)
        set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no DIPU device available, return
        if cur_stream is None or self.idx == -1:
            return
        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream) -> StreamContext:
    return StreamContext(stream)

class Event(_C._DIPUEventBase):
    r"""Wrapper around a dipu event.

    dipu events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize dipu
    streams.

    The underlying dipu events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    """

    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, self).__init__(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch_dipu.dipu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_dipu.dipu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super(Event, self).elapsed_time(end_event)

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``dipuEventSynchronize()``: see
            `dipu Event documentation`_ for more info.
        """
        super(Event, self).synchronize()
    

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.dipu_event)

    def __repr__(self):
        if self.dipu_event:
            return '<torch_dipu.dipu.Event {0:#x}>'.format(self._as_parameter_.value)
        else:
            return '<torch_dipu.dipu.Event uninitialized>'
