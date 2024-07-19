import os
import sys

sys.path.append(os.getenv("PYTORCH_DIR", default=""))
import torch
from pathlib import Path

print(torch._C._PYBIND11_BUILD_ABI[-2:])
print(Path(torch.__path__[0]).parent.absolute())
print(int(torch.compiled_with_cxx11_abi()))
print(torch.utils.cmake_prefix_path)
print(int(torch.cuda.is_available()))
