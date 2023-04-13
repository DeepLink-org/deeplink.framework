import os
import sys
import subprocess
import multiprocessing
import functools
import signal
import torch

from time import sleep
from ctypes import cdll
from threading import Thread
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, _compile_start, _compile_end
from torch._inductor.codecache import TritonFuture, CppCodeCache, _worker_compile, _load_kernel
from torch._inductor import config, cuda_properties, exc
from torch.hub import _Faketqdm, tqdm
from typing import Any, Dict


class AscendCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        picked_vec_isa = pick_vec_isa()
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa),
        )

        output_path = input_path[:-3] + 'so'
        output_graph_path = os.path.split(output_path)[0] + '/graph'
        if key not in cls.cache:
            #if not os.path.exists(output_path) or True:
            cmd = ['/usr/bin/c++',
                   '-D_GLIBCXX_USE_CXX11_ABI=0',
                   '-fPIC',
                   '-shared',
                   '-std=c++11',
                   '-g',
                   '-Wall',
                   '-I/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc',
                   '-I/usr/local/Ascend/ascend-toolkit/latest/include/graph',
                   '-I/usr/local/Ascend/ascend-toolkit/latest/include/ge',
                   '-I/usr/local/Ascend/ascend-toolkit/latest/parser',
                   '-I/usr/local/Ascend/ascend-toolkit/latest/compiler/include',
                   '-I/daoxin/pytorch/third_party/DICP/AscendGraph/codegen',
                   '-L/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub',
                   '-lgraph',
                   '-lge_runner',
                   '-o' + output_path,
                   input_path,
                   '-Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub',
                   '/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub/libgraph.so',
                   '/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub/libge_runner.so',]
            try:
                subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise exc.CppCompileError(cmd, e.output) from e
            loaded = cdll.LoadLibrary(output_path)
            loaded.compile(output_graph_path.encode())
            
            from codegen.load_and_run import AscendExecutor
            exe = AscendExecutor(0, output_graph_path + '.om')
            cls.cache[key] = exe
            cls.cache[key].key = key
        return cls.cache[key]

class AsyncCompileAscend:
    def __init__(self):
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool():
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool():
        # ensure properties have been calculated before processes
        # are forked
        cuda_properties._properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()

        # if this process dies abnormally (e.g. segfault)
        # it will not shut down the workers. Instead
        # the workers will have their parent reassigned to the
        # init process. This launches a separate thread to
        # watch for the worker getting reassigned,
        # and cleans it up in this case.
        def init():
            def run():
                while True:
                    sleep(1)
                    if orig_ppid != os.getppid():
                        os.kill(os.getpid(), signal.SIGKILL)

            global _watchdog_thread
            _watchdog_thread = Thread(target=run, daemon=True)
            _watchdog_thread.start()

        # we rely on 'fork' because we cannot control whether users
        # have an `if __name__ == '__main__'` in their main process.
        fork_context = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(
            config.compile_threads, mp_context=fork_context, initializer=init
        )
        # when this pool is created in a subprocess object, the normal exit handler
        # doesn't run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers...
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    @classmethod
    def warm_pool(cls):
        if config.compile_threads <= 1:
            return
        _compile_start()
        pool = cls.process_pool()

        # We have to fork processes for compiler workers, but the more memory and other resources that are loaded, the
        # slower the os.fork time is, quite drastically. It also holds the GIL so we can't put it on another thread.

        # Examples:
        # A simple x + x + x script: 10ms seconds in the middle of the program, 2ms at startup
        # tf_efficientnet_b0 benchmark: 50ms! in the middle of the program , 3ms at startup

        # So we want to start the workers early when it is still cheap, and also to allow the workers to get
        # ready before we have work for them.

        # ProcessPoolExecutor also does not launch the workers until it finds a point when all the workers are idle.
        # But if we waited until then fork time will be long and we will be waiting for the processes to initialize.

        # We force them to start here with some YOLOing of the internal methods.
        if hasattr(pool, "_start_queue_management_thread"):
            pool._start_queue_management_thread()
        else:
            for _ in range(config.compile_threads):
                pool._adjust_process_count()
            pool._start_executor_manager_thread()
        _compile_end()

    @classmethod
    def submit(cls, task):
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def map(cls, fn, seq):
        if config.compile_threads <= 1 or len(seq) <= 1:
            return list(map(fn, seq))
        return [t.result() for t in [cls.pool().submit(fn, x) for x in seq]]

    def triton(self, source_code):
        _compile_start()

        if config.compile_threads > 1:
            major, minor = torch.cuda.get_device_capability()
            device = torch.cuda.current_device()
            cc = major * 10 + minor
            future = self.process_pool().submit(
                _worker_compile, source_code, cc, device
            )
            return TritonFuture(source_code, future)
        else:
            return _load_kernel(source_code)

    def cpp(self, source_code):
        def task():
            return CppCodeCache.load(source_code).kernel

        return self.submit(task)

    def ascend(self, source_code):
        return AscendCodeCache.load(source_code).run

    def wait(self, scope: Dict[str, Any]):
        num_kernels = len(
            [
                value
                for key, value in scope.items()
                if isinstance(value, (Future, TritonFuture))
            ]
        )
        pbar = tqdm(
            total=num_kernels,
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        if config.compile_threads > 1:
            for key, result in scope.items():
                if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                    pbar.set_postfix_str(key)
                if isinstance(result, (Future, TritonFuture)):
                    scope[key] = result.result()
                    pbar.update(1)

        _compile_end()

