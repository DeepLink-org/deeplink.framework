#include "graph_utils.h"

extern "C" int compile(char* graph_path, char* graph_json_file) {
  std::map<AscendString, AscendString> options;
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());

  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);
  buildGraph(graph, graph_json);

  bool has_dynamic_shape = graph_json["has_dynamic_shape"].get<bool>();
  if (has_dynamic_shape) {
    for (const auto& item : graph_json["build_options"]) {
      auto key = item["name"].get<std::string>();
      auto value = item["value"].get<std::string>();
      options.insert({AscendString(key.c_str()), AscendString(value.c_str())});
    }
  }

  AclgraphBuilder builder;
  builder.saveGraph(graph_path, graph, options);
  return SUCCESS;
}