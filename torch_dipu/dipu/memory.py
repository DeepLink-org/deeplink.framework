
import collections

from torch_dipu import _C
from .device import current_device, _get_device_index
from .utils import is_initialized
from .streams import current_stream, Stream


def caching_allocator_alloc(size, device=None, stream=None):
    r"""Performs a memory allocation using the dipu memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch_dipu.dipu.caching_allocator_delete`.

    Arguments:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default dipu device is used.
        stream (torch_dipu.dipu.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`dipu-memory-management` for more details about dipu memory
        management.
    """
    if device is None:
        device = current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = current_stream(device)
    if isinstance(stream, Stream):
        stream = stream.dipu_stream
    if not isinstance(stream, int):
        raise TypeError('Invalid type for stream argument, must be '
                        '`torch_dipu.dipu.Stream` or `int` representing a pointer '
                        'to a exisiting stream')
    with device(device):
        return _C._dipu_dipuCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Deletes memory allocated using the dipu memory allocator.

    Memory allocated with :func:`~torch_dipu.dipu.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Arguments:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`dipu-memory-management` for more details about dipu memory
        management.
    """
    _C._dipu_dipuCachingAllocator_raw_delete(mem_ptr)


def empty_cache():
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other dipu application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch_dipu.dipu.empty_cache` doesn't increase the amount of dipu
        memory available for PyTorch. However, it may help reduce fragmentation
        of dipu memory in certain cases. See :ref:`dipu-memory-management` for
        more details about dipu memory management.
    """
    if is_initialized():
        _C._dipu_emptyCache()


## just an empty shell now
def memory_stats(device=None):
    result = []
    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    ## stats = memory_stats_as_nested_dict(device=device)
    ## _recurse_add_to_result("", stats)
    result.sort()
    return collections.OrderedDict(result)


def _create_metrics_to_display() :
    def _format_size(sz, pref_sz):
        prefixes = ["B ", "KB", "MB", "GB", "TB", "PB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return "{:7d} {}".format(sz, prefix)

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return "{:7d} {} ".format(cnt, prefix)

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("reserved_bytes", "dipu reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "dipu reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch dipu memory summary, device ID {device:<18d} ")
    lines.append("-" * 75)
    lines.append("  {_:9} dipu OOMs: {num_ooms:<13d} | {_:6} dipuMalloc retries: {num_alloc_retries:<9d}  ")
    lines.append("=" * 75)
    lines.append("        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  ")
    return metrics_to_display, lines

def memory_summary(device=None, abbreviated=False):
    r"""Returns a human-readable printout of the current memory allocator
    statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch_dipu.dipu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`dipu-memory-management` for more details about dipu memory
        management.
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)
    metrics_to_display, lines = _create_metrics_to_display()

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = None, None, None, None

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(" {:<21} | {} | {} | {} | {} ".format(
                submetric_name,
                formatter(current, current_prefval),
                formatter(peak, peak_prefval),
                formatter(allocated, allocated_prefval),
                formatter(freed, freed_prefval)),
            )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"

