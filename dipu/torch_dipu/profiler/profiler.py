import os
import torch
from torch_dipu import _C
from operator import attrgetter
import math
from torch.autograd import DeviceType
from torch.autograd.profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    FunctionEvent,
    MEMORY_EVENT_NAME,
    MemRecordsAcc,
    OUT_OF_MEMORY_EVENT_NAME,
    _format_time,
    _format_time_share,
    EventList,
    _format_memory,
)

def dipu_kineto_available():
    return True

class TorchProfile(torch.autograd.profiler.profile):
    def _parse_kineto_results(self, result):
        # result.events() has most of the events - PyTorch op-level and device-level events

        trace_start_us = result.trace_start_us()
        mem_records = [[evt, False] for evt in result.events() if evt.name() == MEMORY_EVENT_NAME]
        oom_records = [evt for evt in result.events() if evt.name() == OUT_OF_MEMORY_EVENT_NAME]
        mem_records_acc = MemRecordsAcc(mem_records)

        def _cpu_memory_usage(mem_record):
            return mem_record.nbytes() if \
                mem_record.device_type() in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP] \
                else 0

        def _cuda_memory_usage(mem_record):
            return mem_record.nbytes() if \
                mem_record.device_type() in [DeviceType.CUDA, DeviceType.HIP] \
                else 0

        # Create and return FunctionEvent list
        function_events = []
        cuda_corr_map: Dict[int, List[FunctionEvent]] = {}
        max_evt_id = 0
        for kineto_event in result.events():
            if _filter_name(kineto_event.name()):
                continue
            rel_start_us = kineto_event.start_us() - trace_start_us
            rel_end_us = rel_start_us + kineto_event.duration_us()
            abs_end_us = kineto_event.start_us() + kineto_event.duration_us()

            cpu_memory_usage = 0
            cuda_memory_usage = 0
            if kineto_event.device_type() == DeviceType.CPU:
                # find the corresponding memory allocation events
                for mem_record in mem_records_acc.in_interval(kineto_event.start_us(), abs_end_us):
                    cpu_memory_usage += _cpu_memory_usage(mem_record[0])
                    cuda_memory_usage += _cuda_memory_usage(mem_record[0])
                    mem_record[1] = True

            is_async = kineto_event.is_async() or (
                kineto_event.start_thread_id() != kineto_event.end_thread_id()
            )

            fe = FunctionEvent(
                id=kineto_event.correlation_id(),
                name=_rewrite_name(name=kineto_event.name(), with_wildcard=True),
                trace_name=_rewrite_name(name=kineto_event.name(), with_wildcard=False),
                thread=kineto_event.start_thread_id(),
                start_us=rel_start_us,
                end_us=rel_end_us,
                fwd_thread=kineto_event.fwd_thread_id(),
                input_shapes=kineto_event.shapes(),
                stack=[entry for entry in kineto_event.stack() if _filter_stack_entry(entry)],
                scope=kineto_event.scope(),
                cpu_memory_usage=cpu_memory_usage,
                cuda_memory_usage=cuda_memory_usage,
                is_async=is_async,
                sequence_nr=kineto_event.sequence_nr(),
                device_type=kineto_event.device_type(),
                device_index=kineto_event.device_index(),
                flops=kineto_event.flops(),
            )
            max_evt_id = fe.id if fe.id > max_evt_id else max_evt_id
            if fe.device_type == DeviceType.CPU and not fe.is_async:
                # Check if we have CUDA time as a fallback
                cuda_time = kineto_event.cuda_elapsed_us()
                if cuda_time > 0:
                    fe.append_kernel(
                        fe.name,
                        fe.device_index,
                        cuda_time)
                    fe.is_legacy = True
            function_events.append(fe)
            corr_id = kineto_event.linked_correlation_id()
            if corr_id > 0:
                if corr_id not in cuda_corr_map:
                    cuda_corr_map[corr_id] = []
                cuda_corr_map[corr_id].append(fe)

        # associate CUDA kernels and CUDA runtime (CPU) with CPU events
        for fe in function_events:
            if (fe.device_type == DeviceType.CPU and not fe.is_async and
                    fe.id in cuda_corr_map):
                kernel_events = []
                for f_evt in cuda_corr_map[fe.id]:
                    if f_evt.device_type == DeviceType.CUDA:
                        kernel_events.append(f_evt)
                    elif f_evt.device_type == DeviceType.CPU:
                        # make sure that 'thread' of a CPU Kineto (e.g. CUDA Runtime) event is associated
                        # with the 'thread' of the corresponding linked PyTorch event to properly track
                        # parents and children
                        f_evt.thread = fe.thread

                if len(kernel_events) == 0:
                    continue

                kernel_events_sorted = sorted(kernel_events, key=lambda event: event.time_range.start)
                max_end_us = -1
                for event in kernel_events_sorted:
                    if event.time_range.end > max_end_us:
                        fe.append_kernel(
                            event.name,
                            event.device_index,
                            event.time_range.end - event.time_range.start)
                        max_end_us = event.time_range.end

        def createFunctionEventForMemoryEvents(evt):
            rel_start_us = evt.start_us() - trace_start_us
            fe = FunctionEvent(
                id=max_evt_id,
                name=evt.name(),
                trace_name=None,  # not outputting in the trace
                thread=evt.start_thread_id(),
                start_us=rel_start_us,
                end_us=rel_start_us,  # no duration
                fwd_thread=evt.start_thread_id(),
                input_shapes=[],
                stack=[],
                scope=0,  # RecordScope::FUNCTION
                cpu_memory_usage=_cpu_memory_usage(evt),
                cuda_memory_usage=_cuda_memory_usage(evt),
                is_async=False,
                sequence_nr=-1,
                device_type=DeviceType.CPU,
                device_index=0,
            )
            return fe

        # output top-level memory events
        for mem_record in mem_records:
            if not mem_record[1]:
                max_evt_id += 1
                fe = createFunctionEventForMemoryEvents(mem_record[0])
                function_events.append(fe)

        for oom_record in oom_records:
            max_evt_id += 1
            fe = createFunctionEventForMemoryEvents(oom_record)
            function_events.append(fe)

        function_events.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
        return function_events


def dipu_build_table(
        events,
        sort_by=None,
        header=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        with_flops=False,
        profile_memory=False,
        top_level_events_only=False):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_cuda_time = any([event.self_cuda_time_total > 0 for event in events])
    has_cuda_mem = any([event.self_cuda_memory_usage > 0 for event in events])
    has_input_shapes = any(
        [(event.input_shapes is not None and len(event.input_shapes) > 0) for event in events])

    if sort_by is not None:
        events = EventList(sorted(
            events, key=lambda evt: getattr(evt, sort_by), reverse=True
        ), use_cuda=has_cuda_time, profile_memory=profile_memory, with_flops=with_flops)

    name_column_width = max([len(evt.key) for evt in events]) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)

    shapes_column_width = max([len(str(evt.input_shapes)) for evt in events]) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)

    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = max([max([len(entry) for entry in stack]) for stack in stacks]) + 4
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    headers = [
        'Name',
        'Self CPU %',
        'Self CPU',
        'CPU total %',
        'CPU total',
        'CPU time avg',
    ]
    if has_cuda_time:
        headers.extend([
            'Self CUDA',
            'Self CUDA %',
            'CUDA total',
            'CUDA time avg',
        ])
    if profile_memory:
        headers.extend([
            'CPU Mem',
            'Self CPU Mem',
        ])
        if has_cuda_mem:
            headers.extend([
                'CUDA Mem',
                'Self CUDA Mem',
            ])
    headers.append(
        '# of Calls'
    )
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any([evt.node_id != -1 for evt in events])
    if append_node_id:
        headers.append('Node ID')

    # Have to use a list because nonlocal is Py3 only...
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]
    MAX_STACK_ENTRY = 5

    def add_column(padding, text_dir='>'):
        row_format_lst[0] += '{: ' + text_dir + str(padding) + '}' + (' ' * SPACING_SIZE)
        header_sep_lst[0] += '-' * padding + (' ' * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        flop_headers = [
            'FLOPs',
            'KFLOPs',
            'MFLOPs',
            'GFLOPs',
            'TFLOPs',
            'PFLOPs',
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append('Input Shapes')
        add_column(shapes_column_width)

    if has_stack:
        headers.append('Source Location')
        add_column(src_column_width, text_dir='<')

    if with_flops:
        # Auto-scaling of flops header
        raw_flops = []
        for evt in events:
            if evt.flops > 0:
                raw_flops.append(evt.flops)
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append('Total {}'.format(flops_header))
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore[assignment]

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well

    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = 0
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            sum_self_cuda_time_total += evt.self_cuda_time_total

    # Actual printing
    if header is not None:
        append('=' * line_length)
        append(header)
    if top_level_events_only:
        append('=' * line_length)
        append('This report only display top-level ops statistics')
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)

    def trim_path(path, src_column_width):
        if len(path) > src_column_width:
            offset = len(path) - src_column_width
            path = path[offset:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        name = evt.key
        if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
            name = name[:(max_name_column_width - 3)] + "..."
        row_values = [
            name,
            # Self CPU total %, 0 for async events.
            _format_time_share(evt.self_cpu_time_total, sum_self_cpu_time_total),
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %, 0 for async events.
            _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total) if not evt.is_async else 0,
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
        ]
        if has_cuda_time:
            row_values.extend([
                evt.self_cuda_time_total_str,
                # CUDA time total %
                _format_time_share(evt.self_cuda_time_total, sum_self_cuda_time_total),
                evt.cuda_time_total_str,
                evt.cuda_time_str,  # Cuda time avg
            ])
        if profile_memory:
            row_values.extend([
                # CPU Mem Total
                _format_memory(evt.cpu_memory_usage),
                # Self CPU Mem Total
                _format_memory(evt.self_cpu_memory_usage),
            ])
            if has_cuda_mem:
                row_values.extend([
                    # CUDA Mem Total
                    _format_memory(evt.cuda_memory_usage),
                    # Self CUDA Mem Total
                    _format_memory(evt.self_cuda_memory_usage),
                ])
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0:
                row_values.append("--")
            else:
                row_values.append('{0:8.3f}'.format(evt.flops * flops_scale))
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:MAX_STACK_ENTRY]:
                append(row_format.format(*(empty_headers + [trim_path(entry, src_column_width)])))
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append("Self CPU time total: {}".format(_format_time(sum_self_cpu_time_total)))
    if has_cuda_time:
        append("Self CUDA time total: {}".format(_format_time(sum_self_cuda_time_total)))
    return ''.join(result)


def apply_profiler_patch():
    # The data collected by dipu profiler differs significantly from pytorch profiler,
    # making it difficult to align during performance analysis.
    # Reuse pytorch profiler logic on NV, while providing environment variables to switch to dipu profiler.
    if _C.dipu_vendor == 'CUDA' and os.environ.get("FORCE_USE_DIPU_PROFILER", 'False').lower() == 'false' :
        return

    setattr(torch.profiler.profiler, 'kineto_available', dipu_kineto_available)
    setattr(torch.autograd.profiler, 'kineto_available', dipu_kineto_available)
    setattr(torch.autograd.profiler, '_prepare_profiler', _C._prepare_profiler)
    setattr(torch.autograd.profiler, '_enable_profiler', _C._enable_profiler)
    setattr(torch.autograd.profiler, '_disable_profiler', _C._disable_profiler)
    setattr(torch.autograd.profiler, '_kineto_step', _C._kineto_step)
    setattr(torch.autograd.profiler, '_supported_activities', _C._supported_activities)
    setattr(torch.autograd, '_supported_activities', _C._supported_activities)
    setattr(torch.autograd, '_add_metadata_json', _C._add_metadata_json)
    setattr(torch.autograd.profiler_util, '_build_table', dipu_build_table)
    torch.autograd.profiler.profile = TorchProfile


class NativeProfile(object):
    def __init__(self, profiler_result_path="./", with_stack=False, record_shapes=False, profile_memory=False):
        self.result_path = profiler_result_path
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.entered = False
        try:
            os.makedirs(self.result_path, exist_ok=True)
        except Exception:
            raise ValueError("the path of '%s' is invaild." % (self.result_path))

    def __enter__(self):
        if self.entered:
            raise RuntimeError("native profile traces are not reentrant")

        self.entered = True
        _C._enable_profiler_api(self.result_path, self.with_stack, self.record_shapes, self.profile_memory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _C._disable_profiler_api()