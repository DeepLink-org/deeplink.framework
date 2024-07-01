#ifndef DICP_ASCEND_GE_RUNNER_H
#define DICP_ASCEND_GE_RUNNER_H

#include <map>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "ascend_string.h"
#include "ge_api.h"
#include "ge_graph.h"
#include "graph_utils.h"

using namespace ge;

class GEGraphRunner {
 public:
  explicit GEGraphRunner(void* context, int device_id,
                         const std::string& config_file_path)
      : device_id_(device_id),
        config_file_path_(config_file_path),
        session_(nullptr) {
    init(static_cast<aclrtContext>(context));
  }

  GEGraphRunner() : device_id_(0), session_(nullptr) {}

  void init(aclrtContext context) {
    // CALL_FUNC(aclrtSetCurrentContext(context));
    std::map<ge::AscendString, ge::AscendString> config;
    config["ge.exec.deviceId"] = std::to_string(device_id_).c_str();

    auto kSocVersion = aclrtGetSocName();
    config[ge::ir_option::SOC_VERSION] = kSocVersion;

    auto raw_conf = parse_json_to_map(config_file_path_);
    for (const auto& item : raw_conf) {
      config[item.first.c_str()] = item.second.c_str();
    }

    for (const auto& item : config) {
      std::cout << "ge init config: " << item.first.GetString() << " = "
                << item.second.GetString() << std::endl;
    }
    CALL_FUNC(ge::GEInitialize(config));

    std::map<ge::AscendString, ge::AscendString> options;
    session_ = std::make_unique<ge::Session>(options);
  }

  std::shared_ptr<CompiledGraphSummary> addGraph(int graph_id,
                                                 const Graph& graph,
                                                 const std::string& graph_key) {
    std::map<ge::AscendString, ge::AscendString> graph_options = {
        {"ge.graph_key", graph_key.c_str()}};
    CALL_FUNC(session_->AddGraph(graph_id, graph, graph_options));
    CALL_FUNC(session_->CompileGraph(graph_id));
    return session_->GetCompiledGraphSummary(graph_id);
  }

  void runGraphWithStreamAsync(std::shared_ptr<GEGraph>& graph, void* stream) {
    CALL_FUNC(session_->RunGraphWithStreamAsync(graph->get_graph_id(), stream,
                                                graph->get_inputs(),
                                                graph->get_outputs()));
  }

  void setConstMem(int graph_id, const void* const memory, size_t size) {
    CALL_FUNC(session_->SetGraphConstMemoryBase(graph_id, memory, size));
  }

  void setFixedFeatureMem(int graph_id, const void* const memory, size_t size) {
    CALL_FUNC(
        session_->SetGraphFixedFeatureMemoryBase(graph_id, memory, size););
  }

  void setFeatureMem(int graph_id, const void* const memory, size_t size) {
    CALL_FUNC(session_->UpdateGraphFeatureMemoryBase(graph_id, memory, size););
  }

  ~GEGraphRunner() {
    session_.reset();
    auto status = ge::GEFinalize();
    if (status != 0) {
      std::cout << "GEFinalize failed!" << std::endl;
    }
  }

  int getDeviceId() { return device_id_; }

 private:
  int device_id_;
  std::string config_file_path_;
  std::unique_ptr<ge::Session> session_;
  aclrtContext context;
  aclrtStream stream;
};

#endif  // DICP_ASCEND_GE_RUNNER_H
