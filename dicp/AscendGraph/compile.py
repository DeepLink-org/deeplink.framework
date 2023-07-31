import os
import subprocess
import time

from ctypes import cdll
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, AsyncCompile
from torch._inductor import exc


class AscendCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code):
        picked_vec_isa = pick_vec_isa()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa) + 'local_rank' + str(local_rank),
        )
        #output_path = input_path[:-3] + 'so'
        #output_graph_path = input_path[:-4] + '/graph'
        #print('output_path: ', output_graph_path)

        output_path = input_path[:-3] + 'so'
        output_graph_path = os.path.split(output_path)[0] + '/graph'
        from dicp.AscendGraph.codegen import load_and_run
        graph_util_path = load_and_run.__file__.replace('/load_and_run.py', '')
        start = time.time()
        if key not in cls.cache:
            if not os.path.exists(output_path):
                cmd = ['/usr/bin/c++',
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '-fPIC',
                    '-shared',
                    '-std=c++11',
                    '-O0',
                    '-Wall',
                    '-I/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc',
                    '-I/usr/local/Ascend/ascend-toolkit/latest/include/graph',
                    '-I/usr/local/Ascend/ascend-toolkit/latest/include/ge',
                    '-I/usr/local/Ascend/ascend-toolkit/latest/parser',
                    '-I/usr/local/Ascend/ascend-toolkit/latest/compiler/include',
                    '-I{}'.format(graph_util_path),
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
                print('compile time:', time.time() - start)
                loaded = cdll.LoadLibrary(output_path)
                loaded.compile(output_graph_path.encode())

                if not os.path.exists(output_graph_path + '.om'):
                    output_graph_path += '_linux_x86_64'
                assert(os.path.exists(output_graph_path + '.om'))

            from dicp.AscendGraph.codegen.load_and_run import AscendExecutor, AscendModel
            # exe = AscendExecutor(0, dims, output_graph_path + '.om')
            exe = AscendModel(0, output_graph_path + '.om')
            cls.cache[key] = exe
            cls.cache[key].key = key

        return cls.cache[key]

class AsyncCompileAscend(AsyncCompile):
    def ascend(self, source_code):
        return AscendCodeCache.load(source_code).run

