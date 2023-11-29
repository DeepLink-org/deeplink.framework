// Copyright (c) 2023, DeepLink.
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <acl/acl_prof.h>
#include <stdint.h>

#include <ATen/record_function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/util.h>

#include <csrc_dipu/runtime/device/profilerapis.h>
#include <csrc_dipu/vendor/vendorapi.h>

extern "C" aclError aclprofSetStampTagName(void* stamp, const char* tagName,
                                           uint16_t len);
extern "C" aclError aclprofSetStampTraceMessage(void* stamp, const char* msg,
                                                uint32_t msgLen);
extern "C" aclError aclprofSetStampCallStack(void* stamp, const char* callStack,
                                             uint32_t len);

namespace dipu {
namespace devapis {

class AscendProfiler {
 public:
  AscendProfiler(const AscendProfiler&) = delete;
  AscendProfiler& operator=(const AscendProfiler&) = delete;

  // AscendProfiler designed as a singleton
  static AscendProfiler& instance();

  void enableProfiler(const std::string& dump_path, bool call_stack,
                      bool record_shapes, bool profile_memory);
  void disableProfiler();

  std::unique_ptr<at::ObserverContext> startRecordEvent(
      const at::RecordFunction& fn);
  void finishRecordEvent(const at::RecordFunction& fn,
                         at::ObserverContext* context);

 private:
  AscendProfiler() = default;
  void recordCallStack(void* stamp, int64_t sequence_num,
                       at::RecordScope scope);

 private:
  bool enable_ = false;
  aclprofConfig* config_ = nullptr;
  bool call_stack_ = false;
  bool record_shapes_ = false;
  bool profile_memory_ = false;
};

AscendProfiler& AscendProfiler::instance() {
  static AscendProfiler profiler;
  return profiler;
}

void AscendProfiler::enableProfiler(const std::string& dump_path,
                                    bool call_stack, bool record_shapes,
                                    bool profile_memory) {
  if (enable_) {
    DIPU_LOGW("ascend profiler has already enabled");
    return;
  }

  call_stack_ = call_stack;
  record_shapes_ = record_shapes;
  profile_memory_ = profile_memory;
  int32_t device_index = 0;
  DIPU_CALLACLRT(aclrtGetDevice(&device_index));

  uint64_t npu_event = 431;
  uint64_t aicore_metrics = 1;
  static const uint32_t device_num = 1;
  uint32_t device_ids[device_num] = {static_cast<uint32_t>(device_index)};
  aclprofAicoreEvents* events = nullptr;
  config_ = aclprofCreateConfig(
      device_ids, device_num, static_cast<aclprofAicoreMetrics>(aicore_metrics),
      events, npu_event);
  TORCH_CHECK(config_ != nullptr,
              "aclprofCreateConfig fail, device_index = ", device_index,
              "npu_event = ", npu_event, "aicore_metrics = ", aicore_metrics);

  DIPU_CALLACLRT(aclrtSynchronizeDevice());
  DIPU_CALLACLRT(aclprofInit(dump_path.c_str(), dump_path.size()));
  DIPU_CALLACLRT(aclprofStart(config_));

  at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        return AscendProfiler::instance().startRecordEvent(fn);
      },
      [](const at::RecordFunction& fn, at::ObserverContext* ctx) {
        AscendProfiler::instance().finishRecordEvent(fn, ctx);
      }));
  enable_ = true;
}

void AscendProfiler::disableProfiler() {
  if (!enable_) {
    DIPU_LOGW("ascend profiler has already disabled");
    return;
  }

  DIPU_CALLACLRT(aclrtSynchronizeDevice());
  at::clearThreadLocalCallbacks();
  DIPU_CALLACLRT(aclprofStop(config_));
  DIPU_CALLACLRT(aclprofFinalize());
  enable_ = false;
}

struct AscendObserverContext : public at::ObserverContext {
  AscendObserverContext(void* d, uint32_t n) : data(d), id(n) {}

  void* data = nullptr;
  uint32_t id = 0;
};

std::unique_ptr<at::ObserverContext> AscendProfiler::startRecordEvent(
    const at::RecordFunction& fn) {
  if (!enable_) {
    DIPU_LOGW("ascend profiler not enabled, ignore record event");
    return std::unique_ptr<AscendObserverContext>();
  }

  void* stamp = aclprofCreateStamp();
  TORCH_CHECK(stamp != nullptr, "aclprofCreateStamp fail",
              ", error msg = ", aclGetRecentErrMsg());
  static const std::string tag_name = "torch_op";
  DIPU_CALLACLRT(
      aclprofSetStampTagName(stamp, tag_name.c_str(), tag_name.size()));
  DIPU_CALLACLRT(
      aclprofSetStampTraceMessage(stamp, fn.name(), strlen(fn.name())));

  if (call_stack_) {
    recordCallStack(stamp, fn.seqNr(), fn.scope());
  }

  uint32_t range_id = 0;
  DIPU_CALLACLRT(aclprofRangeStart(stamp, &range_id));
  return std::make_unique<AscendObserverContext>(stamp, range_id);
}

void AscendProfiler::recordCallStack(void* stamp, int64_t sequence_num,
                                     at::RecordScope scope) {
  std::string seq_nr = "seq=" + std::to_string(sequence_num);
  std::vector<std::string> py_stack;
  std::string call_stack_data;

  if (scope != at::RecordScope::BACKWARD_FUNCTION) {
    auto cs =
        torch::profiler::impl::prepareCallstack(torch::jit::currentCallstack());
    if (cs.empty()) {
      cs = torch::profiler::impl::prepareCallstack(
          torch::jit::tracer::pythonCallstack());
    }
    py_stack = torch::profiler::impl::callstackStr(cs);
    call_stack_data = torch::profiler::impl::stacksToStr(py_stack, ";");
  } else {
    call_stack_data = seq_nr;
  }

  if (!call_stack_data.empty()) {
    DIPU_CALLACLRT(aclprofSetStampCallStack(stamp, call_stack_data.c_str(),
                                            call_stack_data.size()));
  }
}

void AscendProfiler::finishRecordEvent(const at::RecordFunction& fn,
                                       at::ObserverContext* context) {
  if (!enable_) {
    DIPU_LOGW("ascend profiler not enabled, ignore record event");
    return;
  }

  AscendObserverContext* ctx_ptr = static_cast<AscendObserverContext*>(context);
  DIPU_CALLACLRT(aclprofRangeStop(ctx_ptr->id));
  aclprofDestroyStamp(ctx_ptr->data);
}

void enableProfiler(const std::string& dump_path, bool call_stack,
                    bool record_shapes, bool profile_memory) {
  AscendProfiler::instance().enableProfiler(dump_path, call_stack,
                                            record_shapes, profile_memory);
}

void disableProfiler() { AscendProfiler::instance().disableProfiler(); }

}  // end namespace devapis
}  // end namespace dipu
