import os
import subprocess

from ctypes import cdll
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, AsyncCompile
from torch._inductor import exc


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
            
            from .codegen.load_and_run import AscendExecutor
            exe = AscendExecutor(0, output_graph_path + '.om')
            cls.cache[key] = exe
            cls.cache[key].key = key
        return cls.cache[key]

class AsyncCompileAscend(AsyncCompile):
    def ascend(self, source_code):
        return AscendCodeCache.load(source_code).run

