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
from skbuild import setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.1"

try:
    import torch
except:
    sys.path.append(os.environ["PYTORCH_DIR"])
    print("PYTHONPATH: " + str(sys.path))
import torch
import builtins


def get_DIPU_ABI_V():
    return next(
        item[-4:-2]
        for item in dir(builtins)
        if "__pybind11_internals_v4_gcc_libstdcpp_cxxabi10" in item
    )


def get_DIPU_COMPILED_WITH_CXX11_ABI():
    return 1 if torch.compiled_with_cxx11_abi() else 0


def get_DIOPI_CMAKE_PREFIX_PATH():
    return torch.utils.cmake_prefix_path


def get_PYTORCH_DIR():
    return os.path.dirname(torch.__path__[0])


def customized_cmake_args():
    cmake_args = list()

    cmake_with_diopi_library = os.getenv("DIPU_WITH_DIOPI_LIBRARY", "INTERNAL")
    cmake_device = os.getenv("DIPU_DEVICE", "cuda")
    cmake_args.append("-DCMAKE_BUILD_TYPE=Release")
    cmake_args.append("-DDEVICE=" + cmake_device)
    cmake_args.append("-DENABLE_COVERAGE=${USE_COVERAGE}")
    cmake_args.append("-DDIPU_ABI_V=" + get_DIPU_ABI_V())
    cmake_args.append(
        "-DDIPU_COMPILED_WITH_CXX11_ABI=" + str(get_DIPU_COMPILED_WITH_CXX11_ABI())
    )
    cmake_args.append("-DDIOPI_CMAKE_PREFIX_PATH=" + get_DIOPI_CMAKE_PREFIX_PATH())
    cmake_args.append("-DPYTORCH_DIR=" + get_PYTORCH_DIR())
    cmake_args.append("-DWITH_DIOPI_LIBRARY=" + cmake_with_diopi_library)
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
    description="DIPU extension for PyTorch",
    url="https://github.com/DeepLink-org/dipu",
    packages=setuptools.find_packages(),
    cmake_args=customized_cmake_args(),
    cmake_install_target="all",
    package_data={
        "torch_dipu": [
            "*.lib",
            "*.so",
            "*.pylib",
            "../third_party/DIOPI/impl/lib/*.lib",
            "../third_party/DIOPI/impl/lib/*.so",
            "../third_party/DIOPI/impl/lib/*.pylib",
        ]
        + torch_dipu_headers(),
    },
    ext_modules=[],
    cmdclass={},
)
