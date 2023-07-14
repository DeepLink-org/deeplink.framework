#include "graph_utils.h"

extern "C" int compile(char* graph_path, char* graph_json_file) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());
  buildGraph(graph, graph_json_file);
  std::map<AscendString, AscendString> options;
  AclgraphBuilder builder;
  builder.saveGraph(graph_path, graph, options);
  return SUCCESS;
}