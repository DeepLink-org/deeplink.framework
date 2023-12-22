#include "graph_utils.h"

static void compile(const std::string& graph_path,
                    const std::string& graph_json_file,
                    const std::string& fusion_switch_file) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());
  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);
  buildGraph(graph, graph_json);

  std::map<AscendString, AscendString> options;
  bool has_dynamic_shape = graph_json["has_dynamic_shape"].get<bool>();
  if (has_dynamic_shape) {
    for (const auto& item : graph_json["build_options"]) {
      auto key = item["name"].get<std::string>();
      auto value = item["value"].get<std::string>();
      options.insert({AscendString(key.c_str()), AscendString(value.c_str())});
    }
  }

  AclgraphBuilder builder{fusion_switch_file};
  builder.saveGraph(graph_path, graph, options);
}

int main(int argc, char* argv[]) {
  std::string graph_path{argv[1]};
  std::string graph_json_file{argv[2]};
  std::string fusion_switch_file{argv[3]};
  compile(graph_path, graph_json_file, fusion_switch_file);
  return 0;
}
