import os
import sys
import setuptools
from skbuild import setup

try:
    import torch
except:
    sys.path.append(os.environ["PYTORCH_DIR"])
    print("PYTHONPATH: " + str(sys.path))
import torch
import builtins


def customized_cmake_args():
    # TODO(refactor): better error checking

    def dipu_abi_v():
        return next(
            item[-4:-2]
            for item in dir(builtins)
            if "__pybind11_internals_v4_gcc_libstdcpp_cxxabi10" in item
        )

    def dipu_compiled_with_cxx11_abi():
        return 1 if torch.compiled_with_cxx11_abi() else 0

    def diopi_cmake_prefix_path():
        return torch.utils.cmake_prefix_path

    def pytorch_dir():
        return os.path.dirname(torch.__path__[0])

    cmake_with_diopi_library = os.getenv("DIPU_WITH_DIOPI_LIBRARY", "INTERNAL")
    cmake_device = os.getenv("DIPU_DEVICE", "cuda")
    cmake_use_coverage = os.getenv("USE_COVERAGE", "OFF")

    return [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DDEVICE={cmake_device}",
        f"-DDIPU_ABI_V={dipu_abi_v()}",
        f"-DDIPU_COMPILED_WITH_CXX11_ABI={dipu_compiled_with_cxx11_abi()}",
        f"-DDIOPI_CMAKE_PREFIX_PATH={diopi_cmake_prefix_path()}",
        f"-DPYTORCH_DIR={pytorch_dir()}",
        f"-DWITH_DIOPI_LIBRARY={cmake_with_diopi_library}",
        f"-DENABLE_COVERAGE={cmake_use_coverage}",
    ]


def torch_dipu_headers():
    return [
        "csrc_dipu/*.h",
        "csrc_dipu/aten/*.h",
        "csrc_dipu/base/*.h",
        "csrc_dipu/binding/*.h",
        "csrc_dipu/diopirt/*.h",
        "csrc_dipu/profiler/*.h",
        "csrc_dipu/utils/*.h",
        "csrc_dipu/runtime/*.h",
        "csrc_dipu/runtime/core/*.h",
        "csrc_dipu/runtime/core/allocator/*.h",
        "csrc_dipu/runtime/core/guardimpl/*.h",
        "csrc_dipu/runtime/device/*.h",
        "csrc_dipu/runtime/devproxy/*.h",
        "csrc_dipu/runtime/distributed/*.h",
        "csrc_dipu/vendor/*/*.h",
    ]


setup(
    name="torch_dipu",
    version="0.4.1",
    description="DIPU extension for PyTorch",
    url="https://github.com/DeepLink-org/dipu",
    packages=setuptools.find_packages(),
    cmake_args=customized_cmake_args(),
    cmake_install_target="all",
    package_data={
        # TODO(lhy,wy): copy DIOPI so file and proto dir to install target dir.
        # Note: only files located in python packages could be copied.
        "torch_dipu": [
            "*.so",
        ]
        + torch_dipu_headers(),
    },
    ext_modules=[],
    cmdclass={},
)
