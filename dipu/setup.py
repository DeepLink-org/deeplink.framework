import os
import sys
import setuptools
from skbuild import setup


def torch_dipu_version():
    return "0.4.1"


def torch_dipu_cmake_args():
    # TODO(refactor): better error checking

    try:
        import torch
    except:
        sys.path.append(os.environ["PYTORCH_DIR"])
        print("PYTHONPATH: " + str(sys.path))
        import torch
    
    dipu_abi_version = torch._C._PYBIND11_BUILD_ABI[-2:]
    dipu_compiled_with_cxx11_abi = int(torch.compiled_with_cxx11_abi())
    diopi_cmake_prefix_path = torch.utils.cmake_prefix_path
    pytorch_dir = os.path.dirname(torch.__path__[0])

    cmake_with_diopi_library = os.getenv("DIPU_WITH_DIOPI_LIBRARY", "INTERNAL")
    cmake_device = os.getenv("DIPU_DEVICE", "cuda")
    cmake_use_coverage = os.getenv("USE_COVERAGE", "OFF")

    return [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DDEVICE={cmake_device}",
        f"-DDIPU_ABI_V={dipu_abi_version}",
        f"-DDIPU_COMPILED_WITH_CXX11_ABI={dipu_compiled_with_cxx11_abi}",
        f"-DDIOPI_CMAKE_PREFIX_PATH={diopi_cmake_prefix_path}",
        f"-DPYTORCH_DIR={pytorch_dir}",
        f"-DWITH_DIOPI_LIBRARY={cmake_with_diopi_library}",
        f"-DENABLE_COVERAGE={cmake_use_coverage}",
    ]


def torch_dipu_public_headers():
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
    version=torch_dipu_version(),
    description="DIPU extension for PyTorch",
    url="https://github.com/DeepLink-org/dipu",
    packages=setuptools.find_packages(),
    cmake_args=torch_dipu_cmake_args(),
    cmake_install_target="all",
    package_data={
        # TODO(diopi): copy DIOPI so file and proto dir to install target dir.
        # Note: only files located in python packages can be copied.
        "torch_dipu": [
            "*.so",
        ]
        + torch_dipu_public_headers(),
    },
    ext_modules=[],
    cmdclass={},
)
