// Copyright (c) 2023, DeepLink.
#include <stdint.h>
#include <iostream>

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <acl/acl_prof.h>

#include <ATen/record_function.h>

#include <csrc_dipu/runtime/device/profilerapis.h>
#include <csrc_dipu/vendor/vendorapi.h>

extern "C" aclError aclprofSetStampTagName(void *stamp, const char *tagName, uint16_t len);
extern "C" aclError aclprofSetStampTraceMessage(void *stamp, const char *msg, uint32_t msgLen);

namespace dipu {
namespace devapis {

class AscendProfiler {
public:
  AscendProfiler(const AscendProfiler &) = delete;
  AscendProfiler &operator=(const AscendProfiler &) = delete;

  // AscendProfiler designed as a singleton
  static AscendProfiler& instance();

  void enableProfiler(const std::string &dump_path, bool call_stack);
  void disableProfiler();

  std::unique_ptr<at::ObserverContext> startRecordEvent(const at::RecordFunction &fn);
  void finishRecordEvent(const at::RecordFunction& fn, at::ObserverContext *context);

private:
  AscendProfiler() = default;

private:
  bool enable_ = false;
  aclprofConfig *config_ = nullptr;
  bool call_stack_ = false;
};

AscendProfiler& AscendProfiler::instance() {
  static AscendProfiler profiler;
  return profiler;
}

void AscendProfiler::enableProfiler(const std::string &dump_path, bool call_stack) {
  if (enable_) {
    return;
  }

  std::cout << "enter into enable profiler" << std::endl;
  call_stack_ = call_stack;
  int32_t device_index = 0;
  DIPU_CALLACLRT(aclrtGetDevice(&device_index));

  // TODO(caikun): how to set npu_event and aicore_metrics??
  uint64_t npu_event = 431;
  uint64_t aicore_metrics = 1;
  static const uint32_t device_num = 1;
  uint32_t device_ids[device_num] = {static_cast<uint32_t>(device_index)};
  aclprofAicoreEvents* events = nullptr;
  config_ = aclprofCreateConfig(
      device_ids, device_num, static_cast<aclprofAicoreMetrics>(aicore_metrics),
      events, npu_event);
  TORCH_CHECK(config_ != nullptr, "aclprofCreateConfig fail, device_index = ", device_index,
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
    return;
  }

  std::cout << "enter into disable profiler" << std::endl;
  DIPU_CALLACLRT(aclrtSynchronizeDevice());
  at::clearThreadLocalCallbacks();
  DIPU_CALLACLRT(aclprofStop(config_));
  DIPU_CALLACLRT(aclprofFinalize());
  enable_ = false;
}

struct AscendObserverContext : public at::ObserverContext {
  AscendObserverContext(void *d, uint32_t n) : data(d), id(n) {}

  void *data = nullptr;
  uint32_t id = 0;
};

// TODO(caikun): how pytorch manager ObserverContext
// TODO(caikun): add call stack option support
std::unique_ptr<at::ObserverContext> AscendProfiler::startRecordEvent(const at::RecordFunction &fn) {
  if (!enable_) {
    return std::unique_ptr<AscendObserverContext>();
  }

  void *stamp = aclprofCreateStamp();
  TORCH_CHECK(stamp != nullptr, "aclprofCreateStamp fail", ", error msg = ", aclGetRecentErrMsg());
  static const std::string tag_name = "torch_op";
  // in /usr/local/Ascend/ascend-toolkit/6.3.RC2.alpha002/x86_64-linux/lib64/libmsprofiler.so
  DIPU_CALLACLRT(aclprofSetStampTagName(stamp, tag_name.c_str(), tag_name.size()));
  DIPU_CALLACLRT(aclprofSetStampTraceMessage(stamp, fn.name(), strlen(fn.name())));
  uint32_t range_id = 0;
  DIPU_CALLACLRT(aclprofRangeStart(stamp, &range_id));
  return std::make_unique<AscendObserverContext>(stamp, range_id);
}

void AscendProfiler::finishRecordEvent(const at::RecordFunction& fn, at::ObserverContext* context) {
  if (!enable_) {
    return;
  }

  AscendObserverContext* ctx_ptr = static_cast<AscendObserverContext*>(context);
  DIPU_CALLACLRT(aclprofRangeStop(ctx_ptr->id));
  aclprofDestroyStamp(ctx_ptr->data);
}

void enableProfiler(const std::string &dump_path, bool call_stack) {
  AscendProfiler::instance().enableProfiler(dump_path, call_stack);
}

void disableProfiler() {
  AscendProfiler::instance().disableProfiler();
}

}  // end namespace devapis
}  // end namespace dipu