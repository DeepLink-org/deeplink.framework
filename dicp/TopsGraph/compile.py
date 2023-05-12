import base64
import dataclasses
import functools
import getpass
import hashlib
import logging
import multiprocessing
import os
import os.path as osp
import re
import shutil
import signal
import subprocess
import sys 
import sysconfig
import tempfile
import types
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import cdll
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict, List

import torch
from torch._inductor.codecache import AsyncCompile
from torch._inductor.codecache import write
from torch._inductor.codecache import cpp_compile_command
from torch._inductor import exc 

class EnflameCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        # picked_vec_isa = pick_vec_isa()
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_compile_command("i", "o"),
        )   
        output_path = input_path[:-3] + 'so'
        codegen_path = osp.join(osp.dirname(osp.abspath(__file__)), "codegen")
        # if True:
        if key not in cls.cache:
            # if True:
            if not osp.exists(output_path):
                cmd = ['/usr/bin/c++', 
                       f'{codegen_path}/src/dtu_utils.cpp',
                       f'{codegen_path}/src/common_ops.cpp', 
                       f'{codegen_path}/src/conv2d_grad.cpp', 
                       f'{codegen_path}/src/maxpool2d_grad.cpp', 
                       '-D_GLIBCXX_USE_CXX11_ABI=0', '-fPIC', '-shared', '-I/usr/include/dtu', 
                       '-I/usr/include/dtu/3_0/runtime', '-L/usr/lib', 
                       f'-I{codegen_path}/include', 
                       '-o' + output_path, input_path, '-ldtu_sdk']
                try:
                    subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    raise exc.CppCompileError(cmd, e.output) from e
            loaded = cdll.LoadLibrary(output_path)
            loaded.compile()
            cls.cache[key] = loaded 
            cls.cache[key].key = key 
        return cls.cache[key]

class AsyncCompileTopsGraph(AsyncCompile):
    def topsgraph(self, source_code):
        return EnflameCodeCache.load(source_code).run
