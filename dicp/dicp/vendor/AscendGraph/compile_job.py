import os
import subprocess
import time

import dicp
from dicp.dynamo_bridge.compile import DeviceCompileJob
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, code_hash
from torch._inductor import exc

from dicp.vendor.AscendGraph.codegen import load_and_run


class AscendCompileJob(DeviceCompileJob):
    def __init__(self, source_code) -> None:
        super().__init__()
        third_party_path = dicp.__file__.replace(
            '/__init__.py', '') + "/third_party"
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
            'local_rank' + str(self._local_rank) + code_hash(compile_file_code)
        )
        self._lib_path = "/tmp/dicp_ascend/ge_graph.so"
        json_util_path = third_party_path + '/nlohmann'
        half_util_path = third_party_path + '/half/include'
        self.fusion_switch_file = graph_util_path + '/fusion_switch.cfg'
        self._cmd = ['/usr/bin/c++',
                     '-D_GLIBCXX_USE_CXX11_ABI=0',
                     '-fPIC',
                     '-std=c++11',
                     '-O3',
                     '-shared',
                     '-Wall',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/include',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/include/graph',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/include/ge',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/parser',
                     '-I/usr/local/Ascend/ascend-toolkit/latest/compiler/include',
                     f'-I{graph_util_path}',
                     f'-I{json_util_path}',
                     f'-I{half_util_path}',
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
                print(' '.join(self._cmd))
                subprocess.check_output(self._cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise exc.CppCompileError(self._cmd, e.output) from e
            print('compile time:', time.time() - start)

    def get_key(self):
        return self._key

    def get_compile_result(self):
        self._compile()
        graph_manager = load_and_run.get_graph_manager()
        current_graph_id = load_and_run.graph_id
        load_and_run.graph_id = load_and_run.graph_id + 1
        graph_manager.add_graph(
            current_graph_id, self._input_path.encode())
        return load_and_run.GEModel(current_graph_id, self._local_rank)

