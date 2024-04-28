#ifndef DICP_ASCEND_GE_RUNNER_H
#define DICP_ASCEND_GE_RUNNER_H

#include <map>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "ascend_string.h"
#include "ge_api.h"

#include "graph_utils.h"

using namespace ge;

class GEGraphRunner {
 public:
  explicit GEGraphRunner(int device_id_, const std::string& config_file_path_) : device_id(device_id_), config_file_path(config_file_path_)
  { 
    init(); 
  }

  explicit GEGraphRunner() {}

  void init() {
    // 1. ge init
    std::map<AscendString, AscendString> config;
    std::string device_id_str = std::to_string(device_id);
    config["ge.exec.deviceId"] = device_id_str.c_str();

    auto kSocVersion = aclrtGetSocName();
    config[ge::ir_option::SOC_VERSION] = kSocVersion;

    auto raw_conf = parse_json_to_map(config_file_path);
    for (const auto& item : raw_conf) {
      config[item.first.c_str()] = item.second.c_str();
    }

    for (const auto& item : config) {
      std::cout << "ge init config: " << item.first.GetString() << " = " << item.second.GetString() << std::endl; 
    }
    CALL_GE_FUNC(ge::GEInitialize(config));

    // 2. add session
    std::map<AscendString, AscendString> options;
    session_ = new Session(options);
    DICP_ASCEND_CHECK_NULLPTR_ABORT(session_);
  }

  std::shared_ptr<CompiledGraphSummary> addGraph(int graph_id,
                                                 const Graph& graph, const std::string& graph_key) {
    std::map<ge::AscendString, ge::AscendString> graph_options = {{"ge.graph_key", graph_key.c_str()}};
    CALL_GE_FUNC(session_->AddGraph(graph_id, graph, graph_options));
    CALL_GE_FUNC(session_->CompileGraph(graph_id));
    return session_->GetCompiledGraphSummary(graph_id);
  }

  void runGraphWithStreamAsync(int graph_id, void* stream,
                               const std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs) {
    CALL_GE_FUNC(session_->RunGraphWithStreamAsync(graph_id, stream, inputs, outputs));
  }

  void setConstMem(int graph_id, const void* const memory, size_t size) {
    CALL_GE_FUNC(session_->SetGraphConstMemoryBase(graph_id, memory, size));
  }

  void setWorkSpace(int graph_id, const void* const memory, size_t size) {
    CALL_GE_FUNC(session_->UpdateGraphFeatureMemoryBase(graph_id, memory, size););
  }

  ~GEGraphRunner() {
    delete session_;
    CALL_GE_FUNC(ge::GEFinalize());
  }

 private:
  int device_id;
  std::string config_file_path;
  ge::Session* session_;
};


#endif  // DICP_ASCEND_GE_RUNNER_H