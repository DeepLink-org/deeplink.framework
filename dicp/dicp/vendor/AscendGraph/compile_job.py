import os
import subprocess
import time
import json
import acl

import torch
import torch_dipu
import dicp
from dicp.dynamo_bridge.compile import DeviceCompileJob
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, code_hash
from torch._inductor import exc

from dicp.vendor.AscendGraph.codegen import load_and_run


class AscendGECompileGERunJob(DeviceCompileJob):
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
        self.device_id = torch_dipu.current_device()
        self.graph = json.loads(source_code.strip())
        self._key, self._input_path = write(
            source_code.strip(),
            "json",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa) +
            str(self.device_id) + code_hash(compile_file_code)
        )
        self._lib_path = "/tmp/dicp_ascend/ge_graph.so"
        json_util_path = third_party_path + '/nlohmann'
        half_util_path = third_party_path + '/half/include'
        self.fusion_switch_file = graph_util_path + '/fusion_switch.cfg'
        self._cmd = ['/usr/bin/c++',
                     '-D_GLIBCXX_USE_CXX11_ABI=0',
                     '-fPIC',
                     '-std=c++17',
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
        context, ret = acl.rt.get_context()
        graph_manager = load_and_run.get_graph_manager()
        current_graph_id = load_and_run.graph_id
        load_and_run.graph_id = load_and_run.graph_id + 1
        graph_key = f'{self._key}_graph{current_graph_id}_device{self.device_id}'

        graph_manager.add_graph(
            current_graph_id, self._input_path.encode(), graph_key.encode())
        ret = acl.rt.set_context(context)
        is_static = not self.graph['has_dynamic_shape']
        if is_static:
            input_nodes = None
            output_nodes = None
        else:
            input_nodes = self.graph['data_nodes']
            output_nodes = self.graph['output_nodes']
        return load_and_run.GEModel(current_graph_id, self.device_id, is_static, input_nodes=input_nodes, output_nodes=output_nodes)


class AscendGECompileAclRunJob(DeviceCompileJob):
    def __init__(self, source_code) -> None:
        super().__init__()
        third_party_path = dicp.__file__.replace(
            '/__init__.py', '') + "/third_party"
        from dicp.vendor.AscendGraph.codegen import load_and_run
        graph_util_path = load_and_run.__file__.replace('/load_and_run.py', '')
        source_path = graph_util_path + '/graph_compile.cpp'
        source_include = graph_util_path + '/graph_utils.h'
        compile_file_code = ''
        for file in [source_path, source_include]:
            with open(file, 'r') as f:
                compile_file_code += f.read()
        picked_vec_isa = pick_vec_isa()
        self.device_id = torch_dipu.current_device()
        self._key, self._input_path = write(
            source_code.strip(),
            "json",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa) +
            str(self.device_id) + code_hash(compile_file_code)
        )
        self._output_graph_path = self._input_path[:-5] + '/graph'
        self._model_path = [f'{self._output_graph_path}.om',
                            f'{self._output_graph_path}_linux_x86_64.om']
        self._lib_path = "/tmp/dicp_ascend/ge_graph.so"
        json_util_path = third_party_path + '/nlohmann'
        half_util_path = third_party_path + '/half/include'
        self.fusion_switch_file = graph_util_path + '/fusion_switch.cfg'
        self.global_options_file = graph_util_path + '/ge_builder_config.json'
        self._cmd = ['/usr/bin/c++',
                     '-D_GLIBCXX_USE_CXX11_ABI=0',
                     '-fPIC',
                     '-std=c++17',
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

    def build_graph(self, output_path, graph_path):
        self._compile()

        graph_compiler = load_and_run.get_graph_compiler()
        graph_compiler.compile_and_save(output_path.encode(), graph_path.encode(
        ), self.fusion_switch_file.encode(), self.global_options_file.encode())

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
        return AscendModel(self.device_id, self._output_graph_path + '.om')
