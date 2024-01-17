import os
import subprocess
import time

from dicp.dynamo_bridge.compile import DeviceCompileJob
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, get_hash
from torch._inductor import exc


class AscendCompileJob(DeviceCompileJob):
    def __init__(self, source_code) -> None:
        super().__init__()
        from dicp.vendor.AscendGraph.codegen import load_and_run
        graph_util_path = load_and_run.__file__.replace('/load_and_run.py', '')
        source_path = graph_util_path + '/graph_compile.cpp'
        source_include = graph_util_path + '/graph_utils.h'
        compile_file_code = ''
        for file in [source_path, source_include]:
            with open(file, 'r') as f:
                compile_file_code += f.read()
        picked_vec_isa = pick_vec_isa()
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._key, self._input_path = write(
            source_code.strip(),
            "json",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa) +
                  'local_rank' + str(self._local_rank) + get_hash(compile_file_code, 'code')
        )
        self._output_graph_path = self._input_path[:-5] + '/graph'
        print('output_path: ', self._output_graph_path)
        self._model_path = [f'{self._output_graph_path}.om',
                            f'{self._output_graph_path}_linux_x86_64.om']
        self._lib_path = "/tmp/dicp_ascend/graph_compile"
        json_util_path = graph_util_path + '/nlohmann'
        self.fusion_switch_file = graph_util_path + '/fusion_switch.cfg'
        self._cmd = ['/usr/bin/c++',
                     '-D_GLIBCXX_USE_CXX11_ABI=0',
                     '-fPIC',
                     '-std=c++11',
                     '-O3',
                     '-Wall',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/include',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/include/graph',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/include/ge',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/parser',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/compiler/include',
                     f'-I{graph_util_path}',
                     f'-I{json_util_path}',
                     '-L/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub',
                     '-lgraph',
                     '-lge_runner',
                     source_path,
                     '-o' + self._lib_path,
                     '/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub/libgraph.so',
                     '/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub/libge_runner.so',
                     '/usr/local/Ascend/ascend-toolkit/latest/lib64/libgraph_base.so',
                     '/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub/libascendcl.so',]

    def _compile(self):
        if not os.path.exists(self._lib_path):
            os.system("mkdir -p /tmp/dicp_ascend")
            start = time.time()
            try:
                subprocess.check_output(self._cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise exc.CppCompileError(self._cmd, e.output) from e
            print('compile time:', time.time() - start)

    def get_key(self):
        return self._key

    def build_graph(self, output_path, graph_path):
        self._compile()
        cmd = [self._lib_path, output_path, graph_path, self.fusion_switch_file]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise exc.CppCompileError(cmd, e.output) from e

    def get_compile_result(self):
        if (not os.path.exists(self._model_path[0]) and not os.path.exists(self._model_path[1])):
            self.build_graph(self._output_graph_path, self._input_path)
        origin_graph_path = self._output_graph_path
        if not os.path.exists(self._output_graph_path + '.om'):
            self._output_graph_path = origin_graph_path + '_linux_x86_64'
        if not os.path.exists(self._output_graph_path + '.om'):
            self._output_graph_path = origin_graph_path + '_linux_aarch64'
        assert (os.path.exists(self._output_graph_path + '.om'))
        from dicp.vendor.AscendGraph.codegen.load_and_run import AscendModel
        return AscendModel(self._local_rank, self._output_graph_path + '.om')
