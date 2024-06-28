#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ge_builder.h"
#include "ge_graph.h"
#include "ge_runner.h"
#include "graph_utils.h"

extern "C" {

std::unordered_map<int, std::shared_ptr<GEGraph>> graph_manager;
std::unique_ptr<GEGraphRunner> graph_runner;

void init(void* context, int device_id, const char* config_file_path) {
  graph_runner =
      std::make_unique<GEGraphRunner>(context, device_id, config_file_path);
  std::cout << "graph runner init success!" << std::endl;
}

void release() { graph_runner.reset(); }

void add_graph(int graph_id, const char* graph_json_file,
               const char* graph_key) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());

  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);

  std::vector<Tensor> input_tensors;
  buildGraph(graph, graph_json, input_tensors);

  auto graph_spec = graph_runner->addGraph(graph_id, graph, graph_key);
  auto acl_graph = std::make_shared<GEGraph>(graph_id, graph_key, graph,
                                             graph_spec, input_tensors);
  graph_manager[graph_id] = std::move(acl_graph);
}

size_t get_const_size(int graph_id) {
  return graph_manager[graph_id]->const_memory_size();
}

size_t get_feature_size(int graph_id) {
  return graph_manager[graph_id]->feature_memory_size();
}

size_t get_fixed_feature_size(int graph_id) {
  return graph_manager[graph_id]->fixed_feature_memory_size();
}

std::string get_shapes(const std::vector<std::vector<int64_t>>& shapes) {
  std::ostringstream oss;
  for (size_t i = 0; i < shapes.size(); ++i) {
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      oss << shapes[i][j] << (j != shapes[i].size() - 1 ? "," : "");
    }
    oss << (i != shapes.size() - 1 ? ";" : "");
  }
  return oss.str();
}

void get_input_shapes(int graph_id, char* input_shapes) {
  std::string str = get_shapes(graph_manager[graph_id]->get_input_shapes());
  strncpy(input_shapes, str.c_str(), str.size());
}

void get_output_shapes(int graph_id, char* output_shapes) {
  std::string str = get_shapes(graph_manager[graph_id]->get_output_shapes());
  strncpy(output_shapes, str.c_str(), str.size());
}

std::string get_dtypes(const std::vector<int>& dtypes) {
  std::ostringstream oss;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    oss << dtypes[i] << (i != dtypes.size() - 1 ? ";" : "");
  }
  return oss.str();
}

void get_input_dtypes(int graph_id, char* input_dtypes) {
  std::string str = get_dtypes(graph_manager[graph_id]->get_input_dtypes());
  strncpy(input_dtypes, str.c_str(), str.size() + 1);
}

void get_output_dtypes(int graph_id, char* output_dtypes) {
  std::string str = get_dtypes(graph_manager[graph_id]->get_output_dtypes());
  strncpy(output_dtypes, str.c_str(), str.size() + 1);
}

void update_inputs(int graph_id, int64_t** shapes, size_t* shape_sizes,
                   size_t outer_size) {
  std::vector<ge::Shape> ge_shapes;
  ge_shapes.reserve(outer_size);
  for (size_t i = 0; i < outer_size; ++i) {
    std::vector<int64_t> inner_shape(shapes[i], shapes[i] + shape_sizes[i]);
    ge_shapes.emplace_back(inner_shape);
  }
  graph_manager[graph_id]->update_inputs(ge_shapes);
}

void update_outputs(int graph_id, int64_t** shapes, size_t* shape_sizes,
                    size_t outer_size) {
  std::vector<ge::Shape> ge_shapes;
  ge_shapes.reserve(outer_size);
  for (size_t i = 0; i < outer_size; ++i) {
    std::vector<int64_t> inner_shape(shapes[i], shapes[i] + shape_sizes[i]);
    ge_shapes.emplace_back(inner_shape);
  }
  graph_manager[graph_id]->update_outputs(ge_shapes);
}

void assemble_inputs(int graph_id, int64_t** shapes, size_t* shape_sizes,
                     size_t outer_size, int* dtypes, int* formats) {
  std::vector<ge::Shape> ge_shapes;
  std::vector<ge::DataType> ge_dtypes;
  std::vector<ge::Format> ge_formats;
  ge_shapes.reserve(outer_size);
  ge_dtypes.reserve(outer_size);
  ge_formats.reserve(outer_size);
  for (size_t i = 0; i < outer_size; ++i) {
    std::vector<int64_t> inner_shape(shapes[i], shapes[i] + shape_sizes[i]);
    ge_shapes.emplace_back(inner_shape);
    ge_dtypes.emplace_back(static_cast<ge::DataType>(dtypes[i]));
    ge_formats.emplace_back(static_cast<ge::Format>(formats[i]));
  }
  graph_manager[graph_id]->assemble_inputs(ge_shapes, ge_dtypes, ge_formats);
}

void assemble_outputs(int graph_id, int64_t** shapes, size_t* shape_sizes,
                      size_t outer_size, int* dtypes, int* formats) {
  std::vector<ge::Shape> ge_shapes;
  std::vector<ge::DataType> ge_dtypes;
  std::vector<ge::Format> ge_formats;
  ge_shapes.reserve(outer_size);
  ge_dtypes.reserve(outer_size);
  ge_formats.reserve(outer_size);
  for (size_t i = 0; i < outer_size; ++i) {
    std::vector<int64_t> inner_shape(shapes[i], shapes[i] + shape_sizes[i]);
    ge_shapes.emplace_back(inner_shape);
    ge_dtypes.emplace_back(static_cast<ge::DataType>(dtypes[i]));
    ge_formats.emplace_back(static_cast<ge::Format>(formats[i]));
  }
  graph_manager[graph_id]->assemble_outputs(ge_shapes, ge_dtypes, ge_formats);
}

void set_graph_memory(int graph_id, void* const_mem_ptr, void* workspace_ptr,
                      size_t const_size, size_t workspace_size) {
  graph_runner->setConstMem(graph_id, const_mem_ptr, const_size);
  graph_runner->setFeatureMem(graph_id, workspace_ptr, workspace_size);
}

void set_fixed_feature_graph_memory(int graph_id, void* workspace_ptr,
                                    size_t workspace_size) {
  graph_runner->setFixedFeatureMem(graph_id, workspace_ptr, workspace_size);
}

void set_feature_graph_memory(int graph_id, void* workspace_ptr,
                              size_t workspace_size) {
  graph_runner->setFeatureMem(graph_id, workspace_ptr, workspace_size);
}

void set_const_graph_memory(int graph_id, void* workspace_ptr,
                            size_t workspace_size) {
  graph_runner->setConstMem(graph_id, workspace_ptr, workspace_size);
}

void run(aclrtContext context, int graph_id, void* stream, void* inputs_data[],
         void* outputs_data[], int64_t inputs_data_size[],
         int64_t outputs_data_size[]) {
  CALL_FUNC(aclrtSetCurrentContext(context));
  graph_manager[graph_id]->set_input_output_data(
      inputs_data, outputs_data, inputs_data_size, outputs_data_size);
  graph_runner->runGraphWithStreamAsync(graph_manager[graph_id], stream);
}

void compile_and_save(const char* graph_path, const char* graph_json_file,
                      const char* fusion_switch_file,
                      const char* global_options_file) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());
  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);

  std::vector<Tensor> input_tensors;
  buildGraph(graph, graph_json, input_tensors);

  std::map<AscendString, AscendString> options;
  bool has_dynamic_shape = graph_json["has_dynamic_shape"].get<bool>();
  if (has_dynamic_shape) {
    for (const auto& item : graph_json["build_options"]) {
      auto key = item["name"].get<std::string>();
      auto value = item["value"].get<std::string>();
      options.insert({AscendString(key.c_str()), AscendString(value.c_str())});
    }
  }

  GEGraphBuilder builder{fusion_switch_file, global_options_file};
  builder.saveGraph(graph_path, graph, options);
}
}
