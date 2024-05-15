import os
import sys

if os.getenv("PYTORCH_DIR", default=None):
    sys.path.append(os.getenv("PYTORCH_DIR", default=None))
import torch
import builtins
from pathlib import Path

print(
    next(
        item[-4:-2]
        for item in dir(builtins)
        if "__pybind11_internals_v4_gcc_libstdcpp_cxxabi10" in item
    )
)
print(Path(torch.__path__[0]).parent.absolute())
print(1 if torch.compiled_with_cxx11_abi() else 0)
print(torch.utils.cmake_prefix_path)
