
fixed_kernel_code = '''from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from third_party.DICP.AscendGraph.compile import AsyncCompileAscend

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompileAscend()

kernel_0 = async_compile.ascend(\'''
#include "graph_utils.h"

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <memory>
#include <numeric>
#include <functional>

uint32_t graph_id = {graph_id};

AscendManager* getAscendManager() {{
    static AscendManager ascendManager(graph_id);
    return &ascendManager;
}}

int32_t genGraph(Graph& graph) {{
    std::vector<Operator> graph_inputs;
    std::vector<Operator> graph_outputs;
{build_graph}
    graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);
    return 0;
}}

extern "C" int compile(char* graph_path) {{
    // 1. Generate graph
    std::string graph_name = "BuildGraph" + graph_id;
    Graph graph(graph_name.c_str());
    Status ret = genGraph(graph);
    if (ret != SUCCESS) {{
        std::cout << "Generate simple graph failed."<<std::endl;
        return FAILED;
    }}
    std::cout<<"Generate simple graph success."<<std::endl;
    
    AclgraphBuilder builder;
    builder.saveGraph(graph_path, graph);
    std::cout << "graph path: " << graph_path << std::endl;
    return SUCCESS;

    // 2. creat session
    auto sess = getAscendManager()->session();

    // 3. add graph
    ret = sess->AddGraph(graph_id, graph);
    if (ret != SUCCESS) {{
        return FAILED;
    }}
    std::cout<<"Session add simple graph success."<<std::endl;
    return SUCCESS;
}}
\''')\n

async_compile.wait(globals())
del async_compile

def call(args):
    ({py_inputs}) = args
    inputs_data = list(map(lambda x: x.data_ptr(), args))
    args.clear()
    output_np = kernel_0(inputs_data)
    {py_outputs}
    {delete_unuse_inputs}
    return ({py_returns})


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    {py_rand_inputs}
    print_performance(lambda: call([{py_inputs}]))
'''


if __name__ == '__main__':
    print(fixed_kernel_code)