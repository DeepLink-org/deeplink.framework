#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "graph_utils.h"

extern "C" {

std::unordered_map<int, std::shared_ptr<AclGraph>> graph_manager;
AclGraphRunner* graph_runner = nullptr;

AclGraphRunner get_graph_runner(int device_id = 0) {
  static AclGraphRunner graph_runner{device_id};
  return graph_runner;
}

AclgraphBuilder get_graph_builder(const std::string& fusion_switch_file = "") {
  AclgraphBuilder builder{fusion_switch_file};
  return builder;
}

void init(int device_id) {
  graph_runner = new AclGraphRunner();
  graph_runner->set_device_id(device_id);
  graph_runner->init();
  std::cout << "graph runner init success!" << std::endl;
}

void release() { delete graph_runner; }

void add_graph(int graph_id, const char* graph_json_file) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());
  std::cout << "graph_json_file: " << graph_json_file << std::endl;
  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);

  std::vector<Tensor> input_tensors;

  buildGraph(graph, graph_json, input_tensors);

  auto acl_graph = std::make_shared<AclGraph>(graph_id, graph);
  acl_graph->inputs = std::move(input_tensors);

  // auto graph_runner = get_graph_runner();
  auto graph_spec =
      graph_runner->addGraph(acl_graph->graph_id_, acl_graph->graph_);
  acl_graph->spec_ = graph_spec;
  acl_graph->prepare_output();
  acl_graph->prapare_input_output_tensordesc();
  graph_manager[graph_id] = acl_graph;
}

size_t get_const_size(int graph_id) {
  return graph_manager[graph_id]->const_memory_size();
}

size_t get_workspace_size(int graph_id) {
  return graph_manager[graph_id]->feature_memory_size();
}

void get_input_shapes(int graph_id, char* input_shapes) {
  auto shapes = graph_manager[graph_id]->get_input_shapes();
  std::ostringstream oss;
  for (unsigned long i = 0; i < shapes.size(); ++i) {
    for (unsigned long j = 0; j < shapes[i].size(); ++j) {
      oss << shapes[i][j];
      if (j != shapes[i].size() - 1) {
        oss << ",";
      }
    }
    if (i != shapes.size() - 1) {
      oss << ";";
    }
  }
  std::string str = oss.str();
  strncpy(input_shapes, str.c_str(), str.size());
}

void get_output_shapes(int graph_id, char* output_shapes) {
  auto shapes = graph_manager[graph_id]->get_output_shapes();
  std::ostringstream oss;
  for (unsigned long i = 0; i < shapes.size(); ++i) {
    for (unsigned long j = 0; j < shapes[i].size(); ++j) {
      oss << shapes[i][j];
      if (j != shapes[i].size() - 1) {
        oss << ",";
      }
    }
    if (i != shapes.size() - 1) {
      oss << ";";
    }
  }
  std::string str = oss.str();
  strncpy(output_shapes, str.c_str(), str.size());
}

void get_input_dtypes(int graph_id, char* input_dtypes) {
  auto dtypes = graph_manager[graph_id]->get_input_dtypes();
  std::ostringstream oss;
  for (unsigned long i = 0; i < dtypes.size(); ++i) {
    oss << dtypes[i];
    if (i != dtypes.size() - 1) {
      oss << ",";
    }
  }
  std::string str = oss.str();
  strncpy(input_dtypes, str.c_str(), str.size());
}

void get_output_dtypes(int graph_id, char* output_dtypes) {
  auto dtypes = graph_manager[graph_id]->get_output_dtypes();
  std::ostringstream oss;
  for (unsigned long i = 0; i < dtypes.size(); ++i) {
    oss << dtypes[i];
    if (i != dtypes.size() - 1) {
      oss << ",";
    }
  }
  std::string str = oss.str();
  strncpy(output_dtypes, str.c_str(), str.size());
}

void set_graph_memory(int graph_id, void* const_mem_ptr, void* workspace_ptr,
                      size_t const_size, size_t workspace_size) {
  graph_runner->setConstMem(graph_id, const_mem_ptr, const_size);
  graph_runner->setWorkSpace(graph_id, workspace_ptr, workspace_size);
}

void run(int graph_id, void* stream, void* inputs_data[], void* outputs_data[],
         int64_t inputs_data_size[], int64_t outputs_data_size[]) {
  graph_manager[graph_id]->set_input_output_data(
      inputs_data, outputs_data, inputs_data_size, outputs_data_size);
  graph_runner->runGraphWithStreamAsync(graph_id, stream,
                                        graph_manager[graph_id]->inputs,
                                        graph_manager[graph_id]->outputs);

  // copy data to output
  for (unsigned int i = 0; i < graph_manager[graph_id]->outputs.size(); ++i) {
    uint8_t* data = graph_manager[graph_id]->outputs[i].GetData();
    auto status = aclrtMemcpy(outputs_data[i], outputs_data_size[i], data,
                              outputs_data_size[i],
                              aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST);
    if (status != 0) {
      std::cout << "aclrtMemcpy failed!" << std::endl;
    }
  }
}
}

