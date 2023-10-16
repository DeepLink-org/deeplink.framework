# Copyright (c) 2023, DeepLink.
import glob
import multiprocessing
import multiprocessing.pool
import os
import re
import shutil
import subprocess
import sys
import platform
import setuptools

import distutils.ccompiler
import distutils.command.clean
from skbuild import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools import distutils, Extension
from setuptools.command.egg_info import egg_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '0.1'

def get_pytorch_dir():
    import torch
    return os.path.dirname(os.path.abspath(torch.__file__))

def start_debug():
    rank = 0
    pid1 = os.getpid()
    print("-------------------------print rank,:", rank, "pid1:", pid1)
    if rank == 0:
        #  time.sleep(15)
        import ptvsd
        host = "127.0.0.1" # or "localhost"
        port = 12345
        print("Waiting for debugger attach at %s:%s ......" % (host, port), flush=True)
        ptvsd.enable_attach(address=(host, port), redirect_output=True)
        ptvsd.wait_for_attach()

# start_debug()

# placeholder: autogen design.....
# generate_bindings_code()

def customized_cmake_args():
    cmake_args = list()
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_args.append("-DDEVICE=cuda")
    cmake_args.append("-DENABLE_COVERAGE=${USE_COVERAGE}")
    cmake_args.append("-DBUILD_DIOPI=TRUE")
    return cmake_args

def torch_dipu_headers():
    headers_pattern = list()
    headers_pattern.append("csrc_dipu/*.h")
    headers_pattern.append("csrc_dipu/aten/*.h")
    headers_pattern.append("csrc_dipu/base/*.h")
    headers_pattern.append("csrc_dipu/binding/*.h")
    headers_pattern.append("csrc_dipu/diopirt/*.h")
    headers_pattern.append("csrc_dipu/profiler/*.h")
    headers_pattern.append("csrc_dipu/utils/*.h")
    headers_pattern.append("csrc_dipu/runtime/*.h")
    headers_pattern.append("csrc_dipu/runtime/core/*.h")
    headers_pattern.append("csrc_dipu/runtime/core/allocator/*.h")
    headers_pattern.append("csrc_dipu/runtime/core/guardimpl/*.h")
    headers_pattern.append("csrc_dipu/runtime/device/*.h")
    headers_pattern.append("csrc_dipu/runtime/devproxy/*.h")
    headers_pattern.append("csrc_dipu/runtime/distributed/*.h")
    headers_pattern.append("csrc_dipu/vendor/*/*.h")
    return headers_pattern

setup(
    name="torch_dipu",
    version=VERSION,
    description='DIPU extension for PyTorch',
    url='https://github.com/DeepLink-org/dipu',
    packages=setuptools.find_packages(),
    cmake_args=customized_cmake_args(),
    cmake_install_target="all",
    package_data = {'torch_dipu':["*.lib","*.so","*.pylib","../third_party/DIOPI/impl/lib/*.lib","../third_party/DIOPI/impl/lib/*.so","../third_party/DIOPI/impl/lib/*.pylib",]+torch_dipu_headers(),},
    ext_modules=[
    ],
    cmdclass={
    })
