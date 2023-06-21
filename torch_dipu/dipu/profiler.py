import torch

from torch.profiler import profile, ProfilerActivity
from torch_dipu import _C

from collections import namedtuple

def add_record_to_pytorch(prof, record_list):
    event_dict = dict()
    event_list = []


    # TODO: calc offset between pytorch start time and dipu start time
    for record in record_list:
        name = record.name.split('(')[0]

        if record.thread_idx >= 90090000:
            event = torch.autograd.profiler_util.FunctionEvent(record.opid, name, record.thread_idx, record.begin / 1000, record.end / 1000, stack=[], device_type=torch.autograd.DeviceType.CUDA, device_index=record.thread_idx, trace_name=name)
            event.duration = event.time_range.elapsed_us()
            event_dict[record.opid] = event
        else:
            event = torch.autograd.profiler_util.FunctionEvent(record.opid, name, record.thread_idx + 100, record.begin / 1000, record.end / 1000, stack=[], device_type=torch.autograd.DeviceType.CPU, device_index=record.thread_idx, trace_name=name, is_legacy=True)
            event_list.append(event)
    for event in event_list:
        event.kernels.append(event_dict[event.id])
        prof.profiler.function_events.append(event)




class dipu_profiler(object):

    def __init__(self, *args, **kwargs):
        kwargs['activities'] = [ProfilerActivity.CPU]
        self._torch_profiler = profile(*args, **kwargs)
        self._running = False
    
    def __enter__(self, *args, **kwargs):
        self._start()
        self._torch_profiler.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self._torch_profiler.__exit__(*args, **kwargs)
        self._stop()

    def start(self, *args, **kwargs):
        self._start()
        self._torch_profiler.start(*args, **kwargs)

    def stop(self, *args, **kwargs):
        self._torch_profiler.stop(*args, **kwargs)
        self._stop()

    def step(self, *args, **kwargs):
        self._torch_profiler.step(*args, **kwargs)
    
    def _start(self):
        if not self._running:
            _C.profile_start()
            self._running = True

    def _stop(self):
        if self._running:
            _C.profiler_flush()
            record_list = _C.get_record()
            add_record_to_pytorch(self._torch_profiler, record_list)

            _C.profile_end()
            self._running = False

    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self._torch_profiler, key)

    def export_chrome_trace(self, path):
        """Exports an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
        import os
        with open(path, 'w') as f:
            chrome_events = []
            next_id = 0
            # Use file IO over using json.dump since JSON dumping is very slow and
            # this technique is proven to give a 4x speedup.
            f.write("[")
            for evt in self._torch_profiler.profiler.function_events:
                if evt.trace_name is None:
                    continue
                f.write(
                    '{"name": "%s", '
                    '"ph": "X", '
                    '"ts": %s, '
                    '"dur": %s, '
                    '"tid": %s, '
                    '"pid": "CPU functions", '
                    '"args": {}}, '
                    % (
                        evt.trace_name,
                        evt.time_range.start,
                        evt.time_range.elapsed_us(),
                        evt.thread
                        if not evt.is_remote
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "'
                        # '"input shape": ' + ('"%s"' % str(evt.input_shapes)),
                        
                    )
                )
                for k in evt.kernels:
                    # 's' and 'f' draw Flow arrows from
                    # the CPU launch to the GPU kernel
                    f.write('{"name": "%s", '
                            '"ph": "s", '
                            '"bp": "e", '
                            '"ts": %s, '
                            '"tid": %s, '
                            '"pid": "CPU functions", '
                            '"id": %s, '
                            '"cat": "cpu_to_cuda", '
                            '"args": {}}, ' % (evt.trace_name, evt.time_range.start,
                                               evt.thread, next_id))
                    f.write(
                        '{"name": "%s", '
                        '"ph": "X", '
                        '"ts": %s, '
                        '"dur": %s, '
                        '"tid": %s, '
                        '"pid": "GPU functions", '
                        '"args": {}}, '
                        % (
                            k.trace_name,
                            k.time_range.start,
                            k.time_range.elapsed_us(),
                            k.thread
                            if not k.is_remote
                            else f'" node_id:{k.node_id}, thread_id:{k.thread} "',
                        )
                    )
                    f.write('{"name": "%s", '
                            '"ph": "f", '
                            '"bp": "e", '
                            '"ts": %s, '
                            '"tid": %s, '
                            '"pid": "GPU functions", '
                            '"id": %s, '
                            '"cat": "cpu_to_cuda", '
                            '"args": {}}, ' % (k.trace_name, k.time_range.start,
                                               k.thread, next_id))
                    # Note: use torch.profiler to get device kernel trace
                    next_id += 1
            if len(self._torch_profiler.profiler.function_events) > 0:
                # remove trailing whitespace and comma
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()
            f.write("]")

def dipu_kineto_available():
    return False
