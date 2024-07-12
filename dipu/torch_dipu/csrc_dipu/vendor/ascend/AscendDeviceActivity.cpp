// Copyright (c) 2023, DeepLink.
#include "AscendDeviceActivity.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <output_base.h>
#include <time_since_epoch.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>

#include <c10/util/Exception.h>

#include "csrc_dipu/utils/Log.h"

namespace dipu {

static const uint64_t kNpuEvents = 431;
static const uint64_t kAicoreMetrics = 1;

using libkineto::GenericTraceActivity;

AscendDeviceActivity::AscendDeviceActivity() {}

AscendDeviceActivity& AscendDeviceActivity::instance() {
  static AscendDeviceActivity instance;
  return instance;
}

// override pure virtual function, do nothing
void AscendDeviceActivity::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::enableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::clearActivities() {}

int32_t AscendDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {

  // 传递USER_ANNOTAION activity标识dump路径
  GenericTraceActivity tmp_path;
  tmp_path.activityName = "random_temp_dir:" + current_dump_path_;
  tmp_path.activityType = libkineto::ActivityType::USER_ANNOTATION;
  tmp_path.startTime = start_time;
  tmp_path.endTime = end_time;
  logger.handleGenericActivity(tmp_path);

  std::string temp_path_prefix = "/tmp/aclprof";
  if (last_dump_path_.compare(0, temp_path_prefix.size(), temp_path_prefix) == 0) {
    // TODO 删除不再使用的last_dump_path_
  }

  return 0;
}

void AscendDeviceActivity::startTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (enable_) {
    DIPU_LOGW("ascend profiler has already enabled");
    return;
  }
  enable_ = true;

  last_dump_path_ = current_dump_path_;

  // 创建随机临时路径
  struct stat st = {0};
  if (stat("/tmp/aclprof", &st) == -1) {
    mkdir("/tmp/aclprof", 0777);
  }
  char dump_path_template[] = "/tmp/aclprof/aclprofXXXXXX";
  char* dump_path_cstring = mkdtemp(dump_path_template);
  if (dump_path_cstring != nullptr) {
    current_dump_path_ = (std::string)dump_path_cstring;
  } else {
    DIPU_LOGE("aclprof random dump path generate failed, the export results may be incorrect");
    current_dump_path_ = "/tmp/aclprof/aclprof_error";
  }

  DIPU_CALLACLRT(aclprofInit(current_dump_path_.c_str(), current_dump_path_.size()));
  DIPU_CALLACLRT(aclrtSynchronizeDevice());

  int32_t device_index = 0;
  DIPU_CALLACLRT(aclrtGetDevice(&device_index));

  std::array<uint32_t, 1> device_ids = {static_cast<uint32_t>(device_index)};
  aclprofAicoreEvents* events = nullptr;
  config_ = aclprofCreateConfig(
      device_ids.data(), device_ids.size(),
      static_cast<aclprofAicoreMetrics>(kAicoreMetrics), events, kNpuEvents);
  TORCH_CHECK(config_ != nullptr,
              "aclprofCreateConfig fail, device_index = ", device_index,
              "npu_event = ", kNpuEvents, "aicore_metrics = ", kAicoreMetrics);

  DIPU_CALLACLRT(aclprofStart(config_));
}

void AscendDeviceActivity::stopTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (!enable_) {
    DIPU_LOGW("ascend profiler has already disabled");
    return;
  }

  DIPU_CALLACLRT(aclrtSynchronizeDevice());
  DIPU_CALLACLRT(aclprofStop(config_));
  DIPU_CALLACLRT(aclprofFinalize());

  enable_ = false;
}


// override pure virtual function, do nothing
void AscendDeviceActivity::teardownContext() {}

// override pure virtual function, do nothing
void AscendDeviceActivity::setMaxBufferSize(int32_t size) {}

// NOLINTNEXTLINE(cppcoreguidelines-interfaces-global-init)
const static int32_t Ascend_device_activity_init = []() {
  const char* env = std::getenv("FORCE_USE_DIPU_PROFILER");
  if ((env == nullptr) || (strncmp(env, "false", strlen("false")) == 0) ||
      (strncmp(env, "False", strlen("False")) == 0)) {
    libkineto::device_activity_singleton = &AscendDeviceActivity::instance();
    return 1;
  }
  return 0;
}();

}  // namespace dipu
