
import os.path as osp
import subprocess
from ctypes import cdll
from torch_dipu.dicp.dynamo_bridge.compile import DeviceCompileJob
from torch._inductor.codecache import write
from torch._inductor.codecache import cpp_compile_command
from torch._inductor import exc 

class TopsCompileJob(DeviceCompileJob):
    def __init__(self, source_code) -> None:
        super().__init__()
        self._key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_compile_command("i", "o"),
        )   
        self._output_path = input_path[:-3] + 'so'
        self._compile_bin_path = input_path[:-3] + 'bin'
        codegen_path = osp.join(osp.dirname(osp.abspath(__file__)), "codegen")
        self._cmd = ['/usr/bin/c++', 
                    '-g', '-O0', '-fPIC', '-shared',
                    '-D_GLIBCXX_USE_CXX11_ABI=0', 
                    f'{codegen_path}/src/dtu_utils.cpp',
                    f'{codegen_path}/src/common_ops.cpp', 
                    f'{codegen_path}/src/conv2d_grad.cpp', 
                    f'{codegen_path}/src/maxpool2d_grad.cpp', 
                    f'-I{codegen_path}/include', 
                    '-I/usr/include/python3.6',
                    '-I/usr/include/dtu/3_0/runtime',
                    '-I/usr/include/dtu',
                    '-L/usr/lib',
                    '-ldtu_sdk',
                    '-o' + self._output_path, input_path]

    def _compile(self):
        try:
            subprocess.check_output(self._cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise exc.CppCompileError(self._cmd, e.output) from e

    def get_key(self):
        return self._key
    
    def get_compile_result(self):
        if not osp.exists(self._output_path):
            self._compile()
        loaded = cdll.LoadLibrary(self._output_path)
        import ctypes
        if not osp.exists(self._compile_bin_path):
            loaded.compile_out(ctypes.c_wchar_p(self._compile_bin_path))
        loaded.load(ctypes.c_wchar_p(self._compile_bin_path))
        return loaded